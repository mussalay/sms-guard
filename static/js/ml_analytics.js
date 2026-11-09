'use strict';

(async function () {
  const errorBox = document.querySelector('#error');
  try {
    if (typeof Chart === 'undefined') throw Error('Chart.js недоступен. Метрики остаются в таблицах.');
    const response = await fetch('/api/v1/ml/analytics/', {credentials: 'same-origin'});
    const report = await response.json();
    if (!response.ok) throw Error(report.error?.message || 'Не удалось прочитать ML-отчёты');
    const names = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc', 'pr_auc'];
    const colors = ['#276b8e', '#198754', '#7b49a0', '#a34b19'];
    const options = (axisLabel) => ({
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {legend: {position: 'bottom'}},
      scales: {
        y: {min: 0, max: 1, title: {display: true, text: 'Значение метрики (0–1)'}},
        x: {title: {display: true, text: axisLabel}}
      }
    });
    const shortName = row => {
      const title = {MultinomialNB: 'NB', LogisticRegression: 'LR', CalibratedLinearSVM: 'SVM'}[row.algorithm] || row.algorithm;
      return row.parameters ? `${title} (${Object.entries(row.parameters).map(([k, v]) => `${k}=${v}`).join(', ')})` : title;
    };
    for (const [kind, model] of Object.entries(report)) {
      new Chart(document.getElementById(`${kind}-metrics`), {
        type: 'bar', data: {labels: names.map(name => name.toUpperCase()), datasets: [
          {label: 'Выбранная модель · validation', data: names.map(n => model.validation[n]), backgroundColor: colors[0]},
          {label: 'Выбранная модель · test', data: names.map(n => model.test[n]), backgroundColor: colors[1]}
        ]}, options: options('Метрика')
      });
      new Chart(document.getElementById(`${kind}-comparison`), {
        type: 'bar', data: {labels: ['PRECISION', 'RECALL', 'F1', 'F2'], datasets: model.comparison.map((row, i) => ({
          label: `${shortName(row)} · validation`, data: ['precision', 'recall', 'f1', 'f2'].map(n => row.validation[n]), backgroundColor: colors[i]
        }))}, options: options('Метрика · сравнение на validation')
      });
      const matrix = model.test.confusion_matrix;
      const points = matrix.flatMap((row, y) => row.map((count, x) => ({x, y, count})));
      const maximum = Math.max(1, ...points.map(p => p.count));
      new Chart(document.getElementById(`${kind}-matrix`), {
        type: 'scatter', data: {datasets: [{data: points, pointRadius: 0, pointHitRadius: 35}]},
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {legend: {display: false}, tooltip: {callbacks: {label: context => `Количество: ${context.raw.count}`}}},
          scales: {
            x: {min: -0.5, max: 1.5, afterBuildTicks: scale => {scale.ticks = [{value: 0}, {value: 1}];}, ticks: {callback: v => model.dataset.labels[v] || ''}, title: {display: true, text: 'Прогноз'}},
            y: {min: -0.5, max: 1.5, reverse: true, afterBuildTicks: scale => {scale.ticks = [{value: 0}, {value: 1}];}, ticks: {callback: v => model.dataset.labels[v] || ''}, title: {display: true, text: 'Факт'}}
          }
        },
        plugins: [{id: 'confusionCells', afterDatasetsDraw(chart) {
          const {ctx, chartArea: area} = chart;
          const width = area.width / 2, height = area.height / 2;
          ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = 'bold 18px sans-serif';
          for (const p of points) {
            const strength = p.count / maximum;
            ctx.fillStyle = `rgba(39,107,142,${0.1 + strength * 0.85})`;
            ctx.fillRect(area.left + p.x * width + 2, area.top + p.y * height + 2, width - 4, height - 4);
            ctx.fillStyle = strength > 0.5 ? '#fff' : '#172a38';
            ctx.fillText(String(p.count), area.left + (p.x + 0.5) * width, area.top + (p.y + 0.5) * height);
          }
          ctx.restore();
        }}]
      });
    }
    document.querySelectorAll('[aria-label="Разделы ML Analytics"] a').forEach(link => {
      link.addEventListener('click', () => {
        document.querySelectorAll('[aria-label="Разделы ML Analytics"] a').forEach(item => item.classList.remove('active'));
        link.classList.add('active');
      });
    });
    document.getElementById('page-data').dataset.chartsReady = 'true';
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove('d-none');
  }
})();
