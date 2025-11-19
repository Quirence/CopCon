from django.db import models

class SensorData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    voltage = models.FloatField(help_text="Напряжение (В)")
    current = models.FloatField(help_text="Сила тока (А)")
    pwm_duty_cycle = models.FloatField(help_text="Скважность ШИМ")
    rpm_percentage = models.FloatField(help_text="Обороты в % от максимума (0-100)")
    flight_id = models.IntegerField(help_text="ID полёта (например, от 1 до 10)")

    def __str__(self):
        return f"Flight {self.flight_id} at {self.timestamp}: V={self.voltage}, I={self.current}"

    class Meta:
        verbose_name = "Sensor Data Entry"
        verbose_name_plural = "Sensor Data Entries"
        ordering = ['-timestamp']