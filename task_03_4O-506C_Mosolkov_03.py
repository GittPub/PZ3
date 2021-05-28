# -*- coding: utf-8 -*-
'''
Гауссов импульс распространяется в одну сторону (TFSF boundary).
Источник находится в диэлектрике.
'''

import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class GaussianMod:
    '''
    Источник, создающий модулированный гауссов импульс
    '''

    def __init__(self, dg, wg, N1, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса,
        N1 - количество отсчетов на длину волны,
        Sc - число Куранта.
        '''
        self.dg = dg
        self.wg = wg
        self.N1 = N1
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        e = (q - m * numpy.sqrt(self.eps * self.mu) / self.Sc - self.dg) / self.wg
        s = numpy.sin(2 * numpy.pi *
                      (q *self.Sc - m * numpy.sqrt(self.eps * self.mu)) / self.N1)
        return self.magnitude * s * numpy.exp(-(e ** 2))
    

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 600

    # Размер области моделирования вдоль оси X в метрах
    X = 1.5

    #Размер ячейки разбиения
    dx = 5e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos + 75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 4.0

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = GaussianMod(100.0, 50.0, 40.0, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / numpy.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    tempRight = Sc / numpy.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Отображение спектра сигнала
    tools.Spectrum(probe.E, dt)

