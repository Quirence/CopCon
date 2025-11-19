import plotly.express as px
import pandas as pd

# Пример данных
data = pd.DataFrame({
    'rpm_percentage': [50, 60, 70, 80, 90],
    'pwm_duty_cycle': [0.4, 0.5, 0.6, 0.7, 0.8],
    'power': [100, 120, 110, 140, 130]  # Потребляемая мощность
})

# Построение 3D графика
fig = px.scatter_3d(data, x='rpm_percentage', y='pwm_duty_cycle', z='power',
                    title='Зависимость мощности от оборотов и ШИМ',
                    labels={'rpm_percentage': 'Обороты (%)', 'pwm_duty_cycle': 'ШИМ', 'power': 'Потребляемая мощность (Вт)'})
fig.show()
