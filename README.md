Определение случаев падения человека с помощью технологий компьютерного зрения
=====
### Описание проекта
В нашем проекте мы используем обученную на нашем датасете модель YoloV8. 
Для Odject Tracking мы воспользовались алгоритмом для отслеживания объектов [SORT](https://github.com/abewley/sort) (A simple online and realtime tracking algorithm), который построен на фильтре Калмана.


### Обучение модели [YoloV8](https://docs.ultralytics.com/)
  1) Из открытых источников мы собрали и разметили датасет состоящий из 4251 изображений. Сбор и разметка изображений происходили на [Roboflow](https://app.roboflow.com)
     ![Nothing](https://cdn.discordapp.com/attachments/1041715072705245236/1174243273787838464/image.png?ex=6566e244&is=65546d44&hm=50f410efc2d2d83541ece257e8f3765c830de9c5fbabb6ef98c892e2c94a2459&).
  2) Обучение происходило на [Google Colabotary](https://colab.google/).
  3) По итогу нескольких обучений с разными параметрами мы получили довольно хорошие результаты.
  ![Nothing](https://cdn.discordapp.com/attachments/1041715072705245236/1174231772989497384/image.png?ex=6566d78e&is=6554628e&hm=8fb239d19de23f2739e68da4f28e72b55ba0c253f8c70d547b998aba6302a1da&).


### Описание работы алгоритма
  1) С помощью нейронной сети YoloV8 программа находит на изображении всех упавших людей.
  2) Каждому из них она присваивает уникальный Id для отслеживания времени нахождения в лежачем состоянии.
  3) После того как программа обнаружила упавшего человека она запоминает время его падения.
  4) Если человек пролежал на полу более 20 секунд программа сообщает об этом оператору.


### Тестирование алгоритма
  [Видео работы](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

### Участники
  1) [Карманов Константин](https://t.me/kostyaka)
  2) [Евдокимов Никита](https://t.me/A102102102102)
  3) [Сафаров Давид](https://t.me/davsf)
  4) [Григорий Попцов](https://t.me/PopcovGrogrij)
