# Métodos de valuación de instrumentos derivados

Este notebook contiene la implementación de algunos de los métodos que son base para la valuación de instrumentos derivados, en particular de opciones, así como un desarrollo de los antecedentes teóricos que los sustentan. Los métodos aquí tratados son:

- Valuación de *opciones europeas* mediante el modelo de **Black-Scholes**

- Valuación de *opciones europeas* mediante el método de **Monte Carlo**

- Valuación de *opciones digitales* mediante **árboles binomiales**

- Valuación de *opciones lookback* mediante la discretización de **Euler-Murayama** y el método de **Monte Carlo**

<br>
<br>

![MonteCarloOptionPricing](/files/image01.png)

___

## Importante

En caso de presentar errores se sugiere probar instalando las versiones de las librerías
 listadas en el archivo ***requirements.txt***, dentro de un ambiente virtual como 
 **venv**, **pipenv**, **virtualvenv** o **conda**.

 ```
 pip install -r requirents.txt
 ```

***Plotters*** es un módulo que contiene funciones auxiliares para graficar
algunos de los resultados obtenidos en los ejemplos planteados en 
**valuacion_instrumentos_derivados.ipynb.** Para poder usarlo, es necesario que el archivo ***plotters.py*** esté en la misma carpeta que el notebook ***valuacion_instrumentos_derivados.ipynb.***
