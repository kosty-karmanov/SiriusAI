Определение случаев падения человека с помощью технологий компьютерного зрения
=====
## Описание проекта:
  ### В нашем проект мы используем обученную на нашем датасете YoloV8. Для Odject Tracking мы успользовали фильтр Калмана который был реализован в [SORT](https://github.com/abewley/sort) (A simple online and realtime tracking algorithm)


## Обучение модели YoloV8:
  ### 1) Из открытых источников мы собрали и разметили датасет состоящий из 4251 изображений
  ### 2) Обучение происходило на [Google Colabotary](https://colab.google/)


### Описание работы алгоритма:
  1) С помощью нейронной сети YoloV8 программа находит на изображении всех упавших людей
  2) Каждому из них она присваивает уникальный Id для отслеживания времяни нахождения в лежачем состоянии
  3) После того как программа обнаружила упавшего человека она запоминает время его падения
  4) Если человек пролежал на полу больше 20 секунд программа сообщает об оператору
