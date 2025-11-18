import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import SensorData

@csrf_exempt  # Отключаем CSRF для упрощения приема POST-запросов из внешнего источника (например, NodeMCU)
def receive_data(request):
    if request.method == 'POST':
        try:
            json_data = json.loads(request.body.decode('utf-8'))

            # Извлечение данных из JSON
            voltage = json_data['voltage']
            current = json_data['current']
            pwm_duty_cycle = json_data['pwm_duty_cycle']
            pwm_frequency = json_data['pwm_frequency']
            pwm_period = json_data['pwm_period']
            percentage_param = json_data['percentage_param']

            # Валидация данных (опционально, но рекомендуется)
            if not (0 <= percentage_param <= 100):
                return JsonResponse({'error': 'percentage_param must be between 0 and 100'}, status=400)

            # Создание и сохранение новой записи в базе данных
            data_entry = SensorData(
                voltage=voltage,
                current=current,
                pwm_duty_cycle=pwm_duty_cycle,
                pwm_frequency=pwm_frequency,
                pwm_period=pwm_period,
                percentage_param=percentage_param
            )
            data_entry.save()

            # Возвращаем успешный ответ
            return JsonResponse({'status': 'success'}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except KeyError as e:
            return JsonResponse({'error': f'Missing key: {str(e)}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        # Возвращаем ошибку, если метод не POST
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def display_data(request):
    # Получаем последнюю запись из базы данных
    latest_data = SensorData.objects.first()
    context = {'latest_data': latest_data}
    return render(request, 'esp/display_data.html', context)