import json
import numpy as np
import pandas as pd
from datetime import timedelta
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib import messages
from django.db.models import Min, Max, Avg
from django.http import JsonResponse, HttpResponseBadRequest
from .models import SensorData
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from django.utils import timezone
import logging
import matplotlib

matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import to_rgba

# Setup logging
logger = logging.getLogger(__name__)

# Define colors that work with matplotlib
GRID_COLOR = (0.392, 0.392, 0.392, 0.2)  # rgba(100,100,100,0.2) converted to 0-1 values
BORDER_COLOR = (0.188, 0.212, 0.235, 1.0)  # #30363d
TEXT_COLOR = (1.0, 1.0, 1.0, 1.0)  # #ffffff
HEADER_BG = (0.129, 0.149, 0.176, 1.0)  # #21262d
CARD_BG = (0.090, 0.106, 0.133, 1.0)  # #161b22
BG_DARK = (0.051, 0.067, 0.090, 1.0)  # #0d1117
ACCENT_COLOR = (0.345, 0.651, 1.0, 1.0)  # #58a6ff
SUCCESS_COLOR = (0.247, 0.725, 0.314, 1.0)  # #3fb950
ERROR_COLOR = (0.973, 0.318, 0.286, 1.0)  # #f85149
WARNING_COLOR = (0.824, 0.600, 0.133, 1.0)  # #d29922


# ---------------------
# Flight Selection View
# ---------------------
@csrf_exempt
def analyze_overview(request):
    """
    Displays flight selection page with:
    - Dropdown of available flights (ID + start time)
    - Refresh button to update flight list
    - Form submission to navigate to detailed analysis
    """
    if request.method == 'POST':
        flight_id = request.POST.get('flight_id')
        if flight_id and flight_id.isdigit():
            return redirect('esp:flight_detail', flight_id=int(flight_id))
        messages.error(request, "Некорректный ID полета")

    # Get unique flights with start times
    flights = SensorData.objects.values('flight_id').annotate(
        start_time=Min('timestamp')
    ).order_by('-start_time')

    flight_options = []
    for flight in flights:
        # Format start time as readable string
        start_time = timezone.localtime(flight['start_time'])
        formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        flight_options.append({
            'id': flight['flight_id'],
            'display': f"{flight['flight_id']} | Начало: {formatted_time}"
        })

    context = {
        'flight_options': flight_options,
        'page_title': "Выбор полета для анализа"
    }
    return render(request, 'esp/flight_selection.html', context)


# ---------------------
# Debug Flight Detail View
# ---------------------
# ... остальной код представлений ...
@csrf_exempt
def flight_detail(request, flight_id):
    """
    Renders detailed analytics for a specific flight with comprehensive debugging information.
    """
    debug_info = {
        'raw_data_stats': {},
        'variability_check': {},
        'model_diagnostics': {},
        'plot_diagnostics': {},
        'errors': [],
        'warnings': []
    }

    try:
        # Get all data for this flight
        data_queryset = SensorData.objects.filter(flight_id=flight_id).order_by('timestamp')
        data_count = data_queryset.count()
        debug_info['data_count'] = data_count

        # Safety check for insufficient data
        if data_count < 10:
            messages.warning(request,
                             f"Для полета #{flight_id} недостаточно данных ({data_count} записей). "
                             "Требуется минимум 10 записей для анализа."
                             )
            debug_info['errors'].append(f"Недостаточно данных: {data_count} < 10")
            return redirect('esp:analyze_overview')

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(list(data_queryset.values(
            'id', 'timestamp', 'voltage', 'current', 'pwm_duty_cycle', 'rpm_percentage'
        )))
        debug_info['raw_data_stats'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': str(df.memory_usage(deep=True).sum()),
            'has_nulls': df.isnull().sum().to_dict(),
            'has_infs': df.isin([np.inf, -np.inf]).sum().to_dict()
        }

        # Calculate power
        df['power'] = df['voltage'] * df['current']

        # Flight duration calculation
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        duration = end_time - start_time
        duration_sec = duration.total_seconds()

        # Basic statistics
        stats = {
            'duration_sec': duration_sec,
            'min_power': df['power'].min(),
            'max_power': df['power'].max(),
            'avg_power': df['power'].mean(),
            'min_voltage': df['voltage'].min(),
            'max_voltage': df['voltage'].max(),
            'avg_voltage': df['voltage'].mean(),
            'min_current': df['current'].min(),
            'max_current': df['current'].max(),
            'avg_current': df['current'].mean(),
        }
        debug_info['basic_stats'] = stats

        # Optimal parameters from raw data (minimum power point)
        min_power_idx = df['power'].idxmin()
        raw_optimum = {
            'power': df.loc[min_power_idx, 'power'],
            'pwm': df.loc[min_power_idx, 'pwm_duty_cycle'],
            'rpm': df.loc[min_power_idx, 'rpm_percentage'],
            'record_id': df.loc[min_power_idx, 'id'],
            'timestamp': df.loc[min_power_idx, 'timestamp']
        }
        debug_info['raw_optimum'] = raw_optimum

        # ---------------------
        # DATA VARIABILITY ANALYSIS
        # ---------------------
        pwm_min, pwm_max = df['pwm_duty_cycle'].min(), df['pwm_duty_cycle'].max()
        rpm_min, rpm_max = df['rpm_percentage'].min(), df['rpm_percentage'].max()
        power_min, power_max = df['power'].min(), df['power'].max()

        pwm_range = pwm_max - pwm_min
        rpm_range = rpm_max - rpm_min
        power_range = power_max - power_min

        variability_info = {
            'pwm': {'min': pwm_min, 'max': pwm_max, 'range': pwm_range, 'std': df['pwm_duty_cycle'].std()},
            'rpm': {'min': rpm_min, 'max': rpm_max, 'range': rpm_range, 'std': df['rpm_percentage'].std()},
            'power': {'min': power_min, 'max': power_max, 'range': power_range, 'std': df['power'].std()}
        }
        debug_info['variability_check'] = variability_info

        # Check minimum variability thresholds
        MIN_PWM_RANGE = 0.01
        MIN_RPM_RANGE = 0.1
        MIN_POWER_RANGE = 0.5

        variability_issues = []
        if pwm_range < MIN_PWM_RANGE:
            variability_issues.append(f"ШИМ диапазон слишком мал: {pwm_range:.6f} < {MIN_PWM_RANGE}")
        if rpm_range < MIN_RPM_RANGE:
            variability_issues.append(f"Обороты диапазон слишком мал: {rpm_range:.2f} < {MIN_RPM_RANGE}")
        if power_range < MIN_POWER_RANGE:
            variability_issues.append(f"Мощность диапазон слишком мал: {power_range:.2f} < {MIN_POWER_RANGE}")

        if variability_issues:
            debug_info['warnings'].extend(variability_issues)
            logger.warning(f"Flight {flight_id} variability issues: {variability_issues}")

        # ---------------------
        # POLYNOMIAL MODEL WITH DEBUGGING
        # ---------------------
        model_optimum = None
        model_diagnostics = {}

        try:
            if not variability_issues:  # Only try model if data is variable enough
                X = df[['pwm_duty_cycle', 'rpm_percentage']].values
                y = df['power'].values

                model_diagnostics.update({
                    'X_shape': X.shape,
                    'y_shape': y.shape,
                    'X_min': X.min(axis=0).tolist(),
                    'X_max': X.max(axis=0).tolist(),
                    'y_min': float(y.min()),
                    'y_max': float(y.max()),
                    'y_std': float(y.std())
                })

                # Create polynomial features (degree=2)
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                model_diagnostics['X_poly_shape'] = X_poly.shape

                # Train model
                model = LinearRegression()
                model.fit(X_poly, y)
                model_diagnostics['model_score'] = model.score(X_poly, y)
                model_diagnostics['model_coefficients_count'] = len(model.coef_)

                # Create grid for predictions
                n_points = 50
                padding_pwm = max(0.01, pwm_range * 0.1)
                padding_rpm = max(0.5, rpm_range * 0.1)

                pwm_grid = np.linspace(max(0, pwm_min - padding_pwm), min(1, pwm_max + padding_pwm), n_points)
                rpm_grid = np.linspace(max(0, rpm_min - padding_rpm), min(100, rpm_max + padding_rpm), n_points)
                pwm_mesh, rpm_mesh = np.meshgrid(pwm_grid, rpm_grid)

                model_diagnostics.update({
                    'grid_shape': pwm_mesh.shape,
                    'pwm_grid_range': [float(pwm_grid.min()), float(pwm_grid.max())],
                    'rpm_grid_range': [float(rpm_grid.min()), float(rpm_grid.max())]
                })

                # Predict power for all grid points
                grid_points = np.column_stack([pwm_mesh.ravel(), rpm_mesh.ravel()])
                grid_poly = poly.transform(grid_points)
                power_pred = model.predict(grid_poly).reshape(pwm_mesh.shape)

                # Check for problematic predictions
                pred_min, pred_max = power_pred.min(), power_pred.max()
                pred_has_nan = np.isnan(power_pred).any()
                pred_has_inf = np.isinf(power_pred).any()

                model_diagnostics.update({
                    'predictions_min': float(pred_min),
                    'predictions_max': float(pred_max),
                    'predictions_range': float(pred_max - pred_min),
                    'predictions_has_nan': pred_has_nan,
                    'predictions_has_inf': pred_has_inf
                })

                if pred_has_nan or pred_has_inf:
                    raise ValueError(f"Предсказания содержат NaN/inf: NaN={pred_has_nan}, Inf={pred_has_inf}")

                # Find minimum power point in predictions
                min_idx = np.unravel_index(np.argmin(power_pred), power_pred.shape)
                model_optimum = {
                    'power': float(power_pred[min_idx]),
                    'pwm': float(pwm_mesh[min_idx]),
                    'rpm': float(rpm_mesh[min_idx]),
                }
                model_diagnostics['model_optimum'] = model_optimum

                # Create STATIC heatmap using matplotlib with CORRECTED COLORS
                plt.figure(figsize=(10, 8), facecolor=CARD_BG)

                # Create the heatmap
                heatmap = plt.contourf(pwm_mesh, rpm_mesh, power_pred,
                                       levels=50, cmap='viridis', alpha=0.9)

                # Add contour lines
                contours = plt.contour(pwm_mesh, rpm_mesh, power_pred,
                                       levels=10, colors='white', alpha=0.5, linewidths=0.5)
                plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

                # Plot the data points
                scatter = plt.scatter(df['pwm_duty_cycle'], df['rpm_percentage'],
                                      c=df['power'], s=30, cmap='viridis',
                                      edgecolors='white', linewidth=0.5, alpha=0.8, zorder=5)

                # Plot the optimal point
                plt.scatter(model_optimum['pwm'], model_optimum['rpm'],
                            color=ERROR_COLOR, s=150, marker='*', edgecolors='white',
                            linewidth=1.5, zorder=10, label=f'Оптимум: {model_optimum["power"]:.1f} Вт')

                # Add labels and title
                plt.xlabel('Скважность ШИМ', color=TEXT_COLOR, fontsize=12)
                plt.ylabel('Обороты (%)', color=TEXT_COLOR, fontsize=12)
                plt.title('Тепловая карта: Распределение мощности', color=TEXT_COLOR, fontsize=14, pad=20)

                # Add colorbar
                cbar = plt.colorbar(heatmap)
                cbar.set_label('Мощность (Вт)', color=TEXT_COLOR, fontsize=10)
                cbar.ax.tick_params(colors=TEXT_COLOR)

                # Customize the plot - CORRECTED GRID COLOR
                plt.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
                plt.legend(facecolor=HEADER_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

                # Set axis colors
                ax = plt.gca()
                ax.set_facecolor(BG_DARK)
                ax.tick_params(colors=TEXT_COLOR)
                ax.spines['bottom'].set_color(BORDER_COLOR)
                ax.spines['left'].set_color(BORDER_COLOR)
                ax.spines['top'].set_color(BORDER_COLOR)
                ax.spines['right'].set_color(BORDER_COLOR)

                # Tight layout
                plt.tight_layout()

                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=CARD_BG)
                plt.close()

                # Encode to base64
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Create the HTML image
                plot_hm = f'<img src="data:image/png;base64,{image_base64}" class="img-fluid" alt="Тепловая карта мощности">'
                debug_info['plot_diagnostics']['heatmap'] = 'Успешно построена с matplotlib (статичная)'

            else:
                # Fallback to simple scatter plot when data variability is insufficient
                debug_info['warnings'].append(
                    "Используется простой scatter plot из-за недостаточной вариативности данных")

                plt.figure(figsize=(10, 8), facecolor=CARD_BG)

                # Create scatter plot
                scatter = plt.scatter(df['pwm_duty_cycle'], df['rpm_percentage'],
                                      c=df['power'], s=60, cmap='viridis',
                                      edgecolors='white', linewidth=1, alpha=0.9)

                # Plot the optimal point
                plt.scatter(raw_optimum['pwm'], raw_optimum['rpm'],
                            color=ERROR_COLOR, s=200, marker='*', edgecolors='white',
                            linewidth=1.5, zorder=10, label=f'Оптимум: {raw_optimum["power"]:.1f} Вт')

                # Add labels and title
                plt.xlabel('Скважность ШИМ', color=TEXT_COLOR, fontsize=12)
                plt.ylabel('Обороты (%)', color=TEXT_COLOR, fontsize=12)
                plt.title('Тепловая карта: Экспериментальные данные', color=TEXT_COLOR, fontsize=14, pad=20)

                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Мощность (Вт)', color=TEXT_COLOR, fontsize=10)
                cbar.ax.tick_params(colors=TEXT_COLOR)

                # Customize the plot - CORRECTED GRID COLOR
                plt.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
                plt.legend(facecolor=HEADER_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

                # Set axis colors
                ax = plt.gca()
                ax.set_facecolor(BG_DARK)
                ax.tick_params(colors=TEXT_COLOR)
                ax.spines['bottom'].set_color(BORDER_COLOR)
                ax.spines['left'].set_color(BORDER_COLOR)
                ax.spines['top'].set_color(BORDER_COLOR)
                ax.spines['right'].set_color(BORDER_COLOR)

                # Tight layout
                plt.tight_layout()

                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=CARD_BG)
                plt.close()

                # Encode to base64
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Create the HTML image
                plot_hm = f'<img src="data:image/png;base64,{image_base64}" class="img-fluid" alt="Тепловая карта данных">'
                debug_info['plot_diagnostics']['heatmap'] = 'Построена как scatter из-за низкой вариативности'

        except Exception as model_error:
            error_msg = f"Ошибка модели: {str(model_error)}"
            debug_info['errors'].append(error_msg)
            logger.error(f"Model error for flight {flight_id}: {error_msg}")

            # Fallback to simple scatter plot
            try:
                plt.figure(figsize=(10, 8), facecolor=CARD_BG)

                # Create scatter plot
                scatter = plt.scatter(df['pwm_duty_cycle'], df['rpm_percentage'],
                                      c=df['power'], s=70, cmap='viridis',
                                      edgecolors='white', linewidth=1, alpha=0.85)

                # Plot the optimal point
                plt.scatter(raw_optimum['pwm'], raw_optimum['rpm'],
                            color=ERROR_COLOR, s=250, marker='*', edgecolors='white',
                            linewidth=1.5, zorder=10, label=f'Оптимум: {raw_optimum["power"]:.1f} Вт')

                # Add labels and title
                plt.xlabel('Скважность ШИМ', color=TEXT_COLOR, fontsize=12)
                plt.ylabel('Обороты (%)', color=TEXT_COLOR, fontsize=12)
                plt.title('Тепловая карта: Данные (аварийный режим)', color=TEXT_COLOR, fontsize=14, pad=20)

                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Мощность (Вт)', color=TEXT_COLOR, fontsize=10)
                cbar.ax.tick_params(colors=TEXT_COLOR)

                # Customize the plot - CORRECTED GRID COLOR
                plt.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
                plt.legend(facecolor=HEADER_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

                # Set axis colors
                ax = plt.gca()
                ax.set_facecolor(BG_DARK)
                ax.tick_params(colors=TEXT_COLOR)
                ax.spines['bottom'].set_color(BORDER_COLOR)
                ax.spines['left'].set_color(BORDER_COLOR)
                ax.spines['top'].set_color(BORDER_COLOR)
                ax.spines['right'].set_color(BORDER_COLOR)

                # Tight layout
                plt.tight_layout()

                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=CARD_BG)
                plt.close()

                # Encode to base64
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Create the HTML image
                plot_hm = f'<img src="data:image/png;base64,{image_base64}" class="img-fluid" alt="Тепловая карта (ошибка)">'
                debug_info['plot_diagnostics']['heatmap'] = f'Fallback после ошибки: {str(model_error)}'

            except Exception as fallback_error:
                plot_hm = f"<div class='alert alert-danger'>Критическая ошибка: {str(model_error)}<br>Fallback: {str(fallback_error)}</div>"
                debug_info['errors'].append(f"Fallback error: {str(fallback_error)}")

        debug_info['model_diagnostics'] = model_diagnostics

        # ---------------------
        # 3D Scatter Plot (FIXED DATA CONVERSION)
        # ---------------------
        try:
            import plotly.graph_objs as go
            from plotly.offline import plot

            fig_3d = go.Figure()

            # ИСПРАВЛЕНО: Явное преобразование pandas Series в списки
            x_data = df['pwm_duty_cycle'].tolist()
            y_data = df['rpm_percentage'].tolist()
            z_data = df['power'].tolist()

            fig_3d.add_trace(go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z_data,  # Используем данные мощности для цвета
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Мощность (Вт)'),
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    'ШИМ: %{x:.3f}<br>'
                    'Обороты: %{y:.1f}%<br>'
                    'Мощность: %{z:.2f} Вт<extra></extra>'
                ),
                showlegend=False
            ))

            # Точка оптимума (здесь f-string допустим)
            fig_3d.add_trace(go.Scatter3d(
                x=[raw_optimum['pwm']],
                y=[raw_optimum['rpm']],
                z=[raw_optimum['power']],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='diamond',
                    color='#ff1744',
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                        '<b>Оптимум</b><br>'
                        'ШИМ: %{x:.3f}<br>'
                        'Обороты: %{y:.1f}%<br>'
                        'Мощность: ' + f'{raw_optimum["power"]:.2f} Вт<extra></extra>'
                ),
                showlegend=False
            ))

            fig_3d.update_layout(
                title='3D: Мощность vs ШИМ и обороты',
                scene=dict(
                    xaxis_title='Скважность ШИМ',
                    yaxis_title='Обороты (%)',
                    zaxis_title='Мощность (Вт)',
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(100,100,100,0.2)'),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(100,100,100,0.2)'),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(100,100,100,0.2)')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff'),
                margin=dict(l=0, r=0, b=0, t=40),
                showlegend=False
            )

            plot_3d = plot(fig_3d, output_type='div', include_plotlyjs=False)
            debug_info['plot_diagnostics']['3d'] = 'Успешно построен'
        except Exception as e:
            # Fallback to simple message
            plot_3d = f"<div class='alert alert-warning text-center py-4'>3D график временно недоступен<br>Ошибка: {str(e)}</div>"
            debug_info['errors'].append(f"3D plot error: {str(e)}")

        # ---------------------
        # PWM vs RPM Scatter Plot (FIXED DATA CONVERSION)
        # ---------------------
        try:
            import plotly.graph_objs as go
            from plotly.offline import plot

            fig_scatter = go.Figure()

            # ИСПРАВЛЕНО: Явное преобразование pandas Series в списки
            x_data = df['pwm_duty_cycle'].tolist()
            y_data = df['rpm_percentage'].tolist()
            power_data = df['power'].tolist()

            # Безопасный расчет размеров точек
            min_power = min(power_data)
            max_power = max(power_data)
            power_range = max_power - min_power if max_power != min_power else 1

            # Расчет размеров для каждой точки (5-20 пикселей)
            sizes = [5 + ((p - min_power) / (power_range + 1e-6) * 15) for p in power_data]

            fig_scatter.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=power_data,
                    colorscale='Viridis',
                    colorbar=dict(title='Мощность (Вт)'),
                    showscale=True,
                    opacity=0.85,
                    line=dict(width=1, color='rgba(255,255,255,0.3)')
                ),
                # ИСПРАВЛЕНО: Используем %{marker.color} вместо жестко заданного значения
                hovertemplate=(
                    'ШИМ: %{x:.3f}<br>'
                    'Обороты: %{y:.1f}%<br>'
                    'Мощность: %{marker.color:.2f} Вт<extra></extra>'
                ),
                showlegend=False
            ))

            # Точка оптимума (здесь f-string допустим)
            fig_scatter.add_trace(go.Scatter(
                x=[raw_optimum['pwm']],
                y=[raw_optimum['rpm']],
                mode='markers',
                marker=dict(
                    size=20,
                    symbol='star-diamond',
                    color='#ff9800',
                    line=dict(width=2, color='#ffffff')
                ),
                hovertemplate=(
                        '<b>Оптимум</b><br>'
                        'ШИМ: %{x:.3f}<br>'
                        'Обороты: %{y:.1f}%<br>'
                        'Мощность: ' + f'{raw_optimum["power"]:.2f} Вт<extra></extra>'
                ),
                showlegend=False
            ))

            fig_scatter.update_layout(
                title='Соотношение ШИМ и оборотов',
                xaxis_title='Скважность ШИМ',
                yaxis_title='Обороты (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(100,100,100,0.2)',
                    range=[max(0, min(x_data) - 0.05), min(1, max(x_data) + 0.05)]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(100,100,100,0.2)',
                    range=[max(0, min(y_data) - 2), min(100, max(y_data) + 2)]
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='closest'
            )

            plot_scatter = plot(fig_scatter, output_type='div', include_plotlyjs=False)
            debug_info['plot_diagnostics']['scatter'] = 'Успешно построен'
        except Exception as e:
            # Fallback to simple message
            plot_scatter = f"<div class='alert alert-warning text-center py-4'>График соотношения временно недоступен<br>Ошибка: {str(e)}</div>"
            debug_info['errors'].append(f"Scatter plot error: {str(e)}")

        # ---------------------
        # Calculate potential efficiency gain
        # ---------------------
        potential_efficiency_gain = None
        if model_optimum and raw_optimum:
            if model_optimum['power'] < raw_optimum['power']:
                potential_efficiency_gain = {
                    'value': ((raw_optimum['power'] - model_optimum['power']) / raw_optimum['power']) * 100,
                    'direction': 'positive'
                }
            elif model_optimum['power'] > raw_optimum['power']:
                potential_efficiency_gain = {
                    'value': ((model_optimum['power'] - raw_optimum['power']) / raw_optimum['power']) * 100,
                    'direction': 'negative'
                }

        # ---------------------
        # Debug Summary
        # ---------------------
        debug_summary = {
            'success': len(debug_info['errors']) == 0,
            'error_count': len(debug_info['errors']),
            'warning_count': len(debug_info['warnings']),
            'data_quality': 'good' if not variability_issues else 'poor',
            'model_quality': model_diagnostics.get('model_score', 'N/A')
        }
        debug_info['summary'] = debug_summary

        # Add debug info to Django messages for user visibility
        if debug_info['warnings']:
            for warning in debug_info['warnings'][:3]:  # Show max 3 warnings
                messages.warning(request, f"Анализ данных: {warning}")

        if debug_info['errors']:
            for error in debug_info['errors'][:2]:  # Show max 2 errors
                messages.error(request, f"Ошибка графика: {error}")

    except Exception as main_error:
        logger.error(f"Critical error in flight_detail for flight {flight_id}: {str(main_error)}")
        messages.error(request, f"Критическая ошибка при анализе полета: {str(main_error)}")
        debug_info['errors'].append(f"Critical error: {str(main_error)}")
        # Return basic context to avoid complete failure
        context = {
            'flight_id': flight_id,
            'data_count': 0,
            'plot_3d': f"<div class='alert alert-danger'>Критическая ошибка: {str(main_error)}</div>",
            'plot_hm': f"<div class='alert alert-danger'>Критическая ошибка: {str(main_error)}</div>",
            'plot_scatter': f"<div class='alert alert-danger'>Критическая ошибка: {str(main_error)}</div>",
            'duration_sec': 0,
            'min_power_raw': 0,
            'opt_pwm_raw': 0,
            'opt_rpm_raw': 0,
            'max_power': 0,
            'avg_power': 0,
            'min_voltage': 0,
            'max_voltage': 0,
            'avg_voltage': 0,
            'min_current': 0,
            'max_current': 0,
            'avg_current': 0,
            'model_opt_pwm': None,
            'model_opt_rpm': None,
            'model_min_power': None,
            'page_title': f"Анализ полета #{flight_id}",
            'debug_info': debug_info  # Include debug info in context
        }
        return render(request, 'esp/flight_detail.html', context)

    # ---------------------
    # Prepare final context
    # ---------------------
    context = {
        'flight_id': flight_id,
        'data_count': data_count,
        'plot_3d': plot_3d,
        'plot_hm': plot_hm,
        'plot_scatter': plot_scatter,
        'duration_sec': duration_sec,
        'min_power_raw': raw_optimum['power'],
        'opt_pwm_raw': raw_optimum['pwm'],
        'opt_rpm_raw': raw_optimum['rpm'],
        'max_power': stats['max_power'],
        'avg_power': stats['avg_power'],
        'min_voltage': stats['min_voltage'],
        'max_voltage': stats['max_voltage'],
        'avg_voltage': stats['avg_voltage'],
        'min_current': stats['min_current'],
        'max_current': stats['max_current'],
        'avg_current': stats['avg_current'],
        'model_opt_pwm': model_optimum['pwm'] if model_optimum else None,
        'model_opt_rpm': model_optimum['rpm'] if model_optimum else None,
        'model_min_power': model_optimum['power'] if model_optimum else None,
        'potential_efficiency_gain': potential_efficiency_gain,
        'page_title': f"Анализ полета #{flight_id}",
        'debug_info': debug_info  # Include debug info in context
    }

    return render(request, 'esp/flight_detail.html', context)


# ---------------------
# Flight Deletion View
# ---------------------
@csrf_exempt
def delete_flight(request, flight_id):
    """
    Handles flight data deletion with password protection.
    Password is hardcoded as 'quirence' per requirements.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Only POST requests are allowed")

    password = request.POST.get('password', '')
    if password != 'quirence':
        messages.error(request, "Неверный пароль! Данные не удалены.")
        return redirect('esp:flight_detail', flight_id=flight_id)

    count, _ = SensorData.objects.filter(flight_id=flight_id).delete()
    messages.success(request, f"Успешно удалено {count} записей для полета #{flight_id}")
    return redirect('esp:analyze_overview')


# ---------------------
# Existing Data Reception View
# ---------------------
@csrf_exempt
def receive_data(request):
    """Handles the incoming flight data from the front-end and saves it to the database."""
    if request.method == 'POST':
        try:
            # Decode and extract the incoming JSON data
            json_data = json.loads(request.body.decode('utf-8'))

            # Extract fields from the data
            voltage = json_data['voltage']
            current = json_data['current']
            pwm_duty_cycle = json_data['pwm_duty_cycle']
            rpm_percentage = json_data['rpm_percentage']
            flight_id = json_data['flight_id']

            # Validate the data
            if not (0 <= rpm_percentage <= 100):
                return JsonResponse({'error': 'rpm_percentage must be between 0 and 100'}, status=400)
            if not isinstance(flight_id, int) or flight_id < 1:
                return JsonResponse({'error': 'flight_id must be a positive integer'}, status=400)

            # Save data to the database
            SensorData.objects.create(
                voltage=voltage,
                current=current,
                pwm_duty_cycle=pwm_duty_cycle,
                rpm_percentage=rpm_percentage,
                flight_id=flight_id
            )

            return JsonResponse({'status': 'success'}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except KeyError as e:
            return JsonResponse({'error': f'Missing key: {str(e)}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)