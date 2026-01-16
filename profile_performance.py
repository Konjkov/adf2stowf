#!/usr/bin/env python3
"""
Пример скрипта для профилирования производительности проекта adf2stowf
"""

import cProfile
import pstats
import io
from adf2stowf.adf2stowf import ADFToStoWF

def profile_adf_conversion():
    """
    Функция для профилирования основного процесса конвертации ADF в StoWF
    """
    # Создаем экземпляр класса с параметрами
    converter = ADFToStoWF(
        plot_cusps=False,
        cusp_method=None,
        do_dump=False,
        cart2harm_projection=True,
        only_occupied=False
    )
    
    # Выполняем основные этапы обработки
    converter.process_valence_basis()
    converter.process_core_basis()
    converter.process_shells()
    converter.process_coefficients()

def main():
    # Создаем профайлер
    pr = cProfile.Profile()
    
    # Запускаем профилирование
    pr.enable()
    profile_adf_conversion()
    pr.disable()
    
    # Создаем строковый поток для вывода результатов
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    
    # Выводим топ 20 самых медленных функций
    ps.print_stats(20)
    
    print("Результаты профилирования:")
    print(s.getvalue())
    
    # Также сохраняем полный отчет в файл
    ps = pstats.Stats(pr)
    ps.sort_stats('cumulative')
    ps.dump_stats('adf2stowf_profile.prof')
    
    print("\nПолный отчет профилирования сохранен в 'adf2stowf_profile.prof'")
    print("Для детального анализа можно использовать:")
    print("python -m pstats adf2stowf_profile.prof")

if __name__ == "__main__":
    main()