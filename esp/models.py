from django.db import models

class SensorData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)  # Автоматически добавляется время создания
    voltage = models.FloatField()  # Текущее напряжение в вольтах
    current = models.FloatField()  # Текущая сила тока в амперах (предположительно, а не вольтах)
    pwm_duty_cycle = models.FloatField()  # Скважность ШИМ (например, в процентах или как дробь)
    pwm_frequency = models.FloatField()  # Частота ШИМ в Гц
    pwm_period = models.FloatField()  # Период ШИМ в секундах
    percentage_param = models.FloatField()  # Процентный параметр (0-100)

    def __str__(self):
        return f"Data at {self.timestamp}: V={self.voltage}, I={self.current}"

    class Meta:
        verbose_name = "Sensor Data Entry"
        verbose_name_plural = "Sensor Data Entries"
        ordering = ['-timestamp'] # Сортировка по убыванию времени (новые первыми)