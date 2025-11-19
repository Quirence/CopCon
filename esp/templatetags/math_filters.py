# esp/templatetags/math_filters.py
from django import template

register = template.Library()

@register.filter
def sub(value, arg):
    """Вычитает arg из value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0