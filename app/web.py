"""Authenticated JSON API and Jinja pages. All user lookups are scoped to the owner."""

import copy
import hmac
import logging
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from app.analysis.services import analyze, mask_text
from app.core.config import BASE_DIR
from app.core.security import hash_password, new_token, token_hash, verify_password
from app.db.models import (
    AuthSession,
    Check,
    DetectedFeature,
    ModelVersion,
    UrlResult,
    User,
    UserProfile,
)

router = APIRouter()
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def fail(status, code, message):
    raise HTTPException(status, detail={"code": code, "message": message})


def db(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def current_user(request: Request, session=Depends(db)):
    token = request.cookies.get("sg_session", "")
    auth = session.get(AuthSession, token_hash(token)) if token else None
    if auth is None or auth.expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
        fail(401, "AUTH_REQUIRED", "Войдите в систему")
    return session.get(User, auth.user_id)


async def data(request: Request):
    try:
        if len(await request.body()) > 16384:
            fail(400, "VALIDATION_ERROR", "Слишком большой запрос")
        value = await request.json()
        if not isinstance(value, dict):
            fail(400, "VALIDATION_ERROR", "Ожидается JSON-объект")
        return value
    except (ValueError, UnicodeError):
        fail(400, "VALIDATION_ERROR", "Ожидается корректный JSON")


def csrf(request: Request):
    expected = request.cookies.get("sg_csrf", "")
    actual = request.headers.get("x-csrf-token", "")
    if not expected or not actual or not hmac.compare_digest(expected, actual):
        fail(403, "CSRF_ERROR", "Обновите страницу и повторите запрос")


def serialize(check):
    created = check.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        **check.payload,
        "id": check.id,
        "created_at": created.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": check.input_text if check.input_text is not None else check.input_masked,
    }


def apply_filters(statement, query):
    for key, column in (
        ("type", Check.check_type),
        ("category", Check.category),
        ("risk_level", Check.risk_level),
    ):
        if query.get(key):
            statement = statement.where(column == query[key])
    for key, operator in (("date_from", "from"), ("date_to", "to")):
        if query.get(key):
            try:
                value = datetime.fromisoformat(query[key])
            except ValueError:
                fail(400, "VALIDATION_ERROR", "Некорректная дата")
            statement = (
                statement.where(Check.created_at >= value)
                if operator == "from"
                else statement.where(Check.created_at < value + timedelta(days=1))
            )
    return statement


@router.post("/api/v1/auth/register/", status_code=201, dependencies=[Depends(csrf)])
async def register(request: Request, session=Depends(db)):
    values = await data(request)
    username = str(values.get("username", "")).strip()
    email = str(values.get("email", "")).strip().lower()
    password = values.get("password", "")
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        fail(400, "VALIDATION_ERROR", "Логин: 3–30 латинских букв, цифр или _")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
        fail(400, "VALIDATION_ERROR", "Введите корректный e-mail")
    if (
        not isinstance(password, str)
        or not 8 <= len(password) <= 128
        or not re.search(r"[A-Za-zА-Яа-я]", password)
        or not re.search(r"\d", password)
    ):
        fail(400, "VALIDATION_ERROR", "Пароль: 8–128 символов, буквы и цифры")
    user = User(
        username=username, email=email, password_hash=hash_password(password), profile=UserProfile()
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        fail(400, "DUPLICATE_USER", "Логин или e-mail уже используется")
    return {"id": user.id, "username": user.username}


@router.post("/api/v1/auth/login/", dependencies=[Depends(csrf)])
async def login(request: Request, session=Depends(db)):
    values = await data(request)
    username = str(values.get("username", ""))
    password = values.get("password", "")
    key = (request.client.host if request.client else "local", username.lower())
    failures = request.app.state.login_failures
    count, blocked = failures.get(key, (0, 0))
    if time.monotonic() < blocked:
        fail(429, "LOGIN_PAUSED", "Слишком много попыток. Подождите 60 секунд")
    user = session.scalar(select(User).where(User.username == username))
    encoded = user.password_hash if user else request.app.state.dummy_hash
    valid = (
        isinstance(password, str) and len(password) <= 128 and verify_password(password, encoded)
    )
    if not user or not valid:
        count += 1
        failures[key] = (count, time.monotonic() + 60 if count >= 5 else 0)
        fail(401, "LOGIN_FAILED", "Неверный логин или пароль")
    failures.pop(key, None)
    old = request.cookies.get("sg_session")
    if old:
        session.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(old)))
    token = new_token()
    session.add(
        AuthSession(
            token_hash=token_hash(token),
            user_id=user.id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
        )
    )
    session.commit()
    response = JSONResponse({"username": user.username})
    response.set_cookie(
        "sg_session",
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=43200,
    )
    return response


@router.post("/api/v1/auth/logout/", status_code=204, dependencies=[Depends(csrf)])
def logout(request: Request, user=Depends(current_user), session=Depends(db)):
    session.execute(
        delete(AuthSession).where(
            AuthSession.token_hash == token_hash(request.cookies["sg_session"])
        )
    )
    session.commit()
    response = Response(status_code=204)
    response.delete_cookie("sg_session")
    return response


async def perform(request, kind, user, session):
    start = time.perf_counter()
    values = await data(request)
    text = values.get("text" if kind == "SMS" else "url", "")
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > (1000 if kind == "SMS" else 2048)
    ):
        fail(400, "VALIDATION_ERROR", "Проверьте ввод: SMS 1–1000, URL 1–2048 символов")
    text = "".join(c for c in text if c in "\n\t" or ord(c) >= 32).strip()
    key = request.headers.get("idempotency-key") or str(uuid4())
    if len(key) > 100:
        fail(400, "VALIDATION_ERROR", "Некорректный идентификатор запроса")
    existing = session.scalar(
        select(Check).where(Check.user_id == user.id, Check.request_key == key)
    )
    if existing:
        return serialize(existing)
    recent = request.app.state.check_limits[user.id]
    now = time.monotonic()
    while recent and recent[0] <= now - 60:
        recent.popleft()
    if len(recent) >= 30:
        fail(429, "RATE_LIMIT", "Не более 30 проверок в минуту")
    recent.append(now)
    try:
        result = analyze(text, kind, request.app.state.models).payload
    except ValueError as error:
        fail(400, "VALIDATION_ERROR", str(error))
    except RuntimeError:
        fail(503, "MODEL_UNAVAILABLE", "Модель недоступна. Обратитесь к администратору")
    result = copy.deepcopy(result)
    if kind == "SMS" and not user.profile.store_sms_text:
        for url in result["urls"]:
            url["url"] = mask_text(url["url"])
            url["domain"] = mask_text(url["domain"])
    check = Check(
        user_id=user.id,
        request_key=key,
        check_type=kind,
        input_text=text if user.profile.store_sms_text else None,
        input_masked=mask_text(text),
        language=result["language"],
        category=result["category"],
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        p_sms=result["p_sms"],
        explanation=result["explanation"],
        recommendation=result["recommendation"],
        sms_model_version=result["model_versions"]["sms"],
        url_model_version=result["model_versions"]["url"],
        payload=result,
    )
    session.add(check)
    session.flush()
    for url in result["urls"]:
        session.add(
            UrlResult(
                check_id=check.id,
                url=url["url"],
                domain=url["domain"],
                p_url=url["p_url"],
                category=url["category"],
                features_json=url["features"],
            )
        )
    for group in result["features"]:
        session.add(DetectedFeature(check_id=check.id, code=group, title=group))
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(
            select(Check).where(Check.user_id == user.id, Check.request_key == key)
        )
        if existing:
            return serialize(existing)
        raise
    logging.getLogger("sms_guard").info(
        "check_id=%s category=%s duration_ms=%.2f",
        check.id,
        check.category,
        1000 * (time.perf_counter() - start),
    )
    return serialize(check)


@router.post("/api/v1/check/sms/", dependencies=[Depends(csrf)])
async def check_sms(request: Request, user=Depends(current_user), session=Depends(db)):
    return await perform(request, "SMS", user, session)


@router.post("/api/v1/check/url/", dependencies=[Depends(csrf)])
async def check_url(request: Request, user=Depends(current_user), session=Depends(db)):
    return await perform(request, "URL", user, session)


@router.get("/api/v1/checks/")
def history(request: Request, user=Depends(current_user), session=Depends(db)):
    try:
        page = max(1, int(request.query_params.get("page", 1)))
    except ValueError:
        fail(400, "VALIDATION_ERROR", "Некорректная страница")
    statement = apply_filters(select(Check).where(Check.user_id == user.id), request.query_params)
    total = session.scalar(select(func.count()).select_from(statement.subquery()))
    checks = session.scalars(
        statement.order_by(Check.created_at.desc(), Check.id.desc())
        .limit(20)
        .offset((page - 1) * 20)
    ).all()
    return {
        "items": [serialize(c) for c in checks],
        "total": total,
        "page": page,
        "pages": max(1, (total + 19) // 20),
    }


@router.get("/api/v1/checks/{check_id}/")
def detail(check_id: int, user=Depends(current_user), session=Depends(db)):
    check = session.scalar(select(Check).where(Check.id == check_id, Check.user_id == user.id))
    if not check:
        fail(404, "NOT_FOUND", "Проверка не найдена")
    return serialize(check)


@router.delete("/api/v1/checks/{check_id}/", status_code=204, dependencies=[Depends(csrf)])
def remove(check_id: int, user=Depends(current_user), session=Depends(db)):
    check = session.scalar(select(Check).where(Check.id == check_id, Check.user_id == user.id))
    if not check:
        fail(404, "NOT_FOUND", "Проверка не найдена")
    session.delete(check)
    session.commit()
    return Response(status_code=204)


@router.get("/api/v1/profile/")
def profile(user=Depends(current_user)):
    return {
        "username": user.username,
        "email": user.email,
        "store_sms_text": user.profile.store_sms_text,
        "preferred_language": user.profile.preferred_language,
        "is_staff": user.is_staff,
    }


@router.post("/api/v1/profile/", dependencies=[Depends(csrf)])
async def update_profile(request: Request, user=Depends(current_user), session=Depends(db)):
    values = await data(request)
    user.profile.store_sms_text = values.get("store_sms_text") is True
    if values.get("password"):
        password = values["password"]
        if not verify_password(str(values.get("current_password", "")), user.password_hash):
            fail(400, "VALIDATION_ERROR", "Неверный текущий пароль")
        if (
            not isinstance(password, str)
            or not 8 <= len(password) <= 128
            or not re.search(r"[A-Za-zА-Яа-я]", password)
            or not re.search(r"\d", password)
        ):
            fail(
                400,
                "VALIDATION_ERROR",
                "Новый пароль должен содержать 8–128 символов, буквы и цифры",
            )
        user.password_hash = hash_password(password)
        session.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    session.commit()
    return {"status": "ok"}


@router.delete("/api/v1/history/", status_code=204, dependencies=[Depends(csrf)])
def clear_history(user=Depends(current_user), session=Depends(db)):
    session.execute(delete(Check).where(Check.user_id == user.id))
    session.commit()
    return Response(status_code=204)


@router.get("/api/v1/stats/summary/")
def stats(request: Request, user=Depends(current_user), session=Depends(db)):
    try:
        days = int(request.query_params.get("days", 30))
    except ValueError:
        fail(400, "VALIDATION_ERROR", "Некорректный период")
    if days not in {7, 30, 90}:
        fail(400, "VALIDATION_ERROR", "Период: 7, 30 или 90 дней")
    conditions = [Check.created_at >= datetime.now(timezone.utc) - timedelta(days=days)]
    if not (user.is_staff and request.query_params.get("scope") == "all"):
        conditions.append(Check.user_id == user.id)
    categories = dict(
        session.execute(
            select(Check.category, func.count()).where(*conditions).group_by(Check.category)
        ).all()
    )
    types = dict(
        session.execute(
            select(Check.check_type, func.count()).where(*conditions).group_by(Check.check_type)
        ).all()
    )
    levels = dict(
        session.execute(
            select(Check.risk_level, func.count()).where(*conditions).group_by(Check.risk_level)
        ).all()
    )
    timeline = session.execute(
        select(func.date(Check.created_at), Check.category, func.count())
        .where(*conditions)
        .group_by(func.date(Check.created_at), Check.category)
        .order_by(func.date(Check.created_at))
    ).all()
    top = session.execute(
        select(DetectedFeature.code, func.count())
        .join(Check, DetectedFeature.check_id == Check.id)
        .where(*conditions)
        .group_by(DetectedFeature.code)
        .order_by(func.count().desc())
        .limit(5)
    ).all()
    latest = session.scalars(
        select(Check)
        .where(*conditions)
        .order_by(Check.created_at.desc(), Check.id.desc())
        .limit(10)
    ).all()
    all_scope = user.is_staff and request.query_params.get("scope") == "all"
    return {
        "total": sum(categories.values()),
        "categories": categories,
        "types": types,
        "levels": levels,
        "average_risk": session.scalar(select(func.avg(Check.risk_score)).where(*conditions)) or 0,
        "timeline": [{"date": d, "category": c, "count": n} for d, c, n in timeline],
        "top_features": dict(top),
        "recent": [] if all_scope else [serialize(c) for c in latest],
        "models": [
            {"type": m.model_type, "version": m.version, "metrics": m.metrics_json}
            for m in session.scalars(select(ModelVersion).where(ModelVersion.is_active))
        ]
        if user.is_staff
        else [],
    }


@router.get("/login/")
@router.get("/register/")
@router.get("/")
@router.get("/check/sms/")
@router.get("/check/url/")
@router.get("/checks/{check_id}/")
@router.get("/history/")
@router.get("/stats/")
@router.get("/profile/")
def page(request: Request, check_id: int | None = None, session=Depends(db)):
    public = request.url.path in {"/login/", "/register/"}
    user = None
    try:
        user = current_user(request, session)
    except HTTPException:
        if not public:
            return RedirectResponse("/login/", status_code=303)
    if check_id is not None:
        detail(check_id, user, session)
    token = request.cookies.get("sg_csrf") or new_token()
    response = templates.TemplateResponse(
        request=request,
        name="app.html",
        context={
            "app_name": request.app.state.settings.app_name,
            "path": request.url.path,
            "user": user,
            "csrf_token": token,
            "check_id": check_id,
        },
    )
    response.set_cookie(
        "sg_csrf",
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=43200,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def initialize_state(application):
    application.state.check_limits = defaultdict(deque)
    application.state.login_failures = {}
    application.state.dummy_hash = hash_password("dummy-invalid-password-000")
