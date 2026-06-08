# Итоговый отчет

Финальная модель `RandomForestRegressor` сравнивается с недельным baseline `SeasonalNaive_168h`.

- `RandomForest`: MAE 1.4310, RMSE 2.3060, SMAPE 2.6122%.
- `SeasonalNaive_168h`: MAE 2.5595, RMSE 4.4690, SMAPE 4.6823%.
- Время обучения `RandomForest`: 27.31 секунд.
- Время инференса на test-периоде: 0.17 секунд.
- Synthetic benchmark для итогового `is_anomaly`: precision 0.82, recall 0.67, F1 0.74.
- Лучший отдельный anomaly-метод на synthetic benchmark: `LOF`, F1 0.79.

Результаты, таблицы и артефакты сохранены в `reports/`. Основной сдаваемый файл: `notebooks/05_final_report.ipynb`.
