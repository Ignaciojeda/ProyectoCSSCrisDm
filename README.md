# ProyectoCSSCriDm
Este es un proyecto con el fin de realizar invesitagaciones para la empresa valve, utilizando un dataset del videojuego Counter Strike: GO

En esta ocasion nos dedicamos a investigar si era posible entrenar varios modelos para responder a 2 preguntas
¿podemos saber si un equipo ganara o perdera la partida solo con datos? y
¿podemos saber cuantas kills tendra el jugador?
Para poder realizar estas predicciones usamos el dataset que nos entrego valve y utilizamos modelos de regresion y clasificacion
como el decision tree, el random forest, el modelo lineal simple y multiple, el suport vector machine y el KNN con estos modelos realizamos diferentes pruebas, las cuales dieron resultados positivos para la investigacion.
Para saber si nuestros modelos eran fiables o no nos guiamos por las metricas de los modelos en el caso del de regresion nos preocupamos que el R cuadrado sea superior a 80 y en el caso del modelo de clasificacion que tuviera una curva ROC superior a 85. Una vez realizadas las pruebas para nuestros proyectos pudimos concluir que los mejores modelos fueron el random forest para ambos modelos.

Una vez listo el entrenamiento y el modelado ahora quedaba desarrollar nuestros modelos para poder utilizar este codigo debes instalar:

-fastapi
-uvicorn
-numpy
-scikit-learn
-pandas

con todas estas librerias cargadas lo unico restante es correr el programa, en la terminal debes poner uvicorn main:app --reload con ese codigo podras visualizar nuestra investigacion, pruebala libremente y sientete libre de hacer los cambios que quieras :).