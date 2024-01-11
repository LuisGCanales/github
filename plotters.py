# plotters.py

# Plotters es un módulo que contiene funciones auxiliares para graficar
# las trayectorias simuladas que se usan en la simulación de precios
# de opciones mediante el método de Montecarlo y el método de árboles binomiales
# implementado en el notebook valuacion_instrumentos_derivados.ipynb.
#
# Para poder usarlo, es necesario que el archivo plotters.py esté en la misma
# carpeta que el notebook valuacion_instrumentos_derivados.ipynb.
 

from matplotlib import pyplot as plt
import networkx
import numpy


def graficar_trayectorias(
    S: numpy.array,
    S0: float,
    r: float,
    T: float,
    dt: float,
    option_type: str,
    payoffs_dict: dict,
    n: int = 100,
) -> None:
    """Grafica "n" trayectorias simuladas del precio del activo subyacente,
    junto con el strike de la opción, la trayectoria formada por el payoff de
    la opción suponiendo fecha de ejercicio t en [0, T], y la evolución de la
    prima de la opción al invertirla en el portafolio replicante para t en
    [0, T].

    El número de trayectorias a graficar, "n", debe ser menor o igual al número
    de simulaciones realizadas.

    Parameters
    ----------
    S : numpy.array
        Arreglo que contiene las trayectorias simuladas del precio del activo
        subyacente.

        S.shape = (n_sim, steps + 1)

        donde n_sim es el número de simulaciones realizadas, y steps es el
        número de pasos en la discretización del tiempo.
    S0 : float
        Precio inicial del activo subyacente.
    r : float
        Tasa libre de riesgo.
    T : float
        Tiempo al vencimiento de la opción, en días.
    dt : float
        Incremento de tiempo usado en la discretización.
    option_type : str
        Tipo de opción ('call' o 'put').
    payoffs_dict : dict
        Diccionario con la función y los parámetros para calcular el payoff de
        la opción, con la siguiente estructura:
            payoff_dict = {
                'func': function
                    Función que calcula los payoffs de la opción sobre un
                    conjunto de trayectorias, al tiempo T, a partir de un objeto
                    numpy.array de dimensiones (n_paths, n_steps), donde
                    'n_paths' es el número de trayectorias y 'n_steps' es el
                    número de pasos de tiempo.
                'params': {
                    'param1': value1,
                    'param2': value2,
                    ...
                    }
                Donde 'param1', 'param2', etc. son los parámetros de la función
                que calcula el payoff de la opción.
                }
    n : int
        Número de trayectorias a graficar.

    Returns
    -------
    None.
    """
    # Se verifica que el número de trayectorias a graficar sea menor o igual al
    # número de simulaciones realizadas.
    error_message = (
        "n debe ser menor o igual al número de trayectorias"
        f" simuladas ({S.shape[0]})."
    )
    assert n <= S.shape[0], error_message

    # Se seleccionan "n" trayectorias aleatorias.
    trayectorias = numpy.random.choice(S.shape[0], size=n, replace=False)

    # Se configura el estilo de las gráficas.
    plt.style.use("ggplot")
    plt.style.use("dark_background")

    # Se inicializan los subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"hspace": 0}
    )

    # Variables auxiliares para la gráfica de las trayectorias.
    steps = S.shape[1]
    time = numpy.linspace(0, T, steps)

    # ---------------------------------------------------------------------
    # Gráfica de las trayectorias simuladas del precio del activo subyacente.
    # ---------------------------------------------------------------------

    # Se grafican las trayectorias seleccionadas.
    for i in trayectorias:
        ax1.plot(time, S[i, :], alpha=0.5)

    # Se grafica el strike.
    K = payoffs_dict["params"]["K"]
    if K is None:
        # Strike flotante
        K = numpy.mean(S[:, -1])
        strike_label = r"Strike flotante $\bar{S}(T)$" + f" = \${K:.2f}"
    else:
        # Strike fijo
        strike_label = r"Strike $K$" + f" = \${K:.2f}"
    
        
    ax1.hlines(
        K,
        xmin=0,
        xmax=T,
        linestyles="dashed",
        color="blue",
        linewidth=3.5,
        label=strike_label,
    )
        

    # ---------------------------------------------------------------------
    # Gráficas de la evolución temporal de la prima y el valor esperado del
    # payoff de la opción.
    # ---------------------------------------------------------------------

    # Se calcula el payoff esperado de la opción para t en [0, T], de esta
    # manera se puede ver cómo cambiaría su valor si la fecha de ejercicio
    # fuera antes.
    payoffs_func = payoffs_dict["func"]
    payoffs_params = payoffs_dict["params"]

    expected_payoffs = []
    for i in range(1, steps + 1):
        payoffs = payoffs_func(S[:, :i], option_type, **payoffs_params)

        # Se estima el valor esperado del payoff de la opción
        expected_payoff = numpy.mean(payoffs)
        expected_payoffs.append(expected_payoff)

    # Se calcula la prima de la opción
    option_price = numpy.exp(-r * T) * expected_payoffs[-1]

    expected_payoffs = numpy.array(expected_payoffs)

    ## Cómo evoluciona el valor de la prima, al invertirla en el portafolio
    ## replicante, es decir, Prima(t) = Prima(0) * exp(rt) para t en [0, T].
    prime_evolution = option_price * numpy.exp(r * numpy.arange(steps) * dt)

    # límites del eje y para ax2
    data_range = prime_evolution[-1] - prime_evolution[0]
    y_min = prime_evolution[0] - 0.1 * data_range
    y_max = prime_evolution[-1] + 0.1 * data_range
    ax2.set_ylim(y_min, y_max)

    # Variable auxiliar para la etiqueta del tipo de opción.
    if option_type == "call":
        opt_sym = "C"
    elif option_type == "put":
        opt_sym = "P"

    # Gráfica de la prima de la opción.
    # ---------------------------------

    prima_label = (
        "$\\text{Prima estimada: }$"
        f"$\hat{{{opt_sym}}}(0) \: = \: \${option_price:.2f}$"
    )
    ax2.scatter(
        0,
        option_price,
        label=prima_label,
        color="lightgreen",
        s=80,
    )

    # Se grafica prima * exp(rt), para t en [0, T].
    # ---------------------------------------------
    prima2_label = "$\hat{" + opt_sym + "}(0) \cdot exp^{rt}$"
    ax2.plot(
        time,
        prime_evolution,
        label=prima2_label,
        linewidth=3,
        color="lightgreen",
    )

    # Gráfica de E[payoff(t) | S(t) > K], para todo t en [0, T].
    # -----------------------------------------------------------

    ep_label = "$\mathbb{\hat{E}}[\, payoff(t) \, | \, S(t) > K \, ]$"
    ax2.plot(time, expected_payoffs, label=ep_label, linewidth=2, color="red")

    # Gráfica de E[payoff(T) | S(T) > K] = Prima(0) * exp(rT).
    # ---------------------------------------------------------
    ep2_label = (
        "$\mathbb{\hat{E}}[ \, payoff(T) \, | \, S(T) > K \, ] \: = \: $"
        "$\hat{" + f"{opt_sym}" + "}(0) \cdot e^{rT} \: = \: \$"
        f"${expected_payoff:.2f}"
    )
    ax2.scatter(
        T,
        expected_payoff,
        label=ep2_label,
        color="red",
        s=200,
        marker=">",
        edgecolors="black",
    )

    # Anotación de la evolución temporal de la prima.
    ax2.annotate(
        f"Evolución temporal de la prima bajo la tasa libre de riesgo",
        xy=(0.1, 0.05),
        xycoords="axes fraction",
        fontsize=14,
        color="lightgreen",
        bbox=dict(boxstyle="round", fc="black", ec="white"),
    )

    # Anotación de la estimación del valor esperado del payoff de la opción.
    ax2.annotate(
        f"Estimación de la esperanza del payoff de la opción para t en [0, T]",
        xy=(0.2, 0.15),
        xycoords="axes fraction",
        fontsize=14,
        color="red",
        bbox=dict(boxstyle="round", fc="black", ec="white"),
    )

    # Anotación con el valor de la tasa libre de riesgo, r.
    ax2.annotate(
        f"$ r = {100 * r:.2f}\%$",
        xy=(0.8, 0.6),
        xycoords="axes fraction",
        fontsize=14,
        color="white",
        bbox=dict(boxstyle="round", fc="black", ec="black"),
    )

    title = (
        "Trayectorias simuladas del precio del activo subyacente para la opción"
        f" tipo {option_type} \n\n Precio inicial del activo subyacente: "
        f"$S_0$ = \${S0:.2f}."
    )

    plt.suptitle(
        title, fontsize=16, y=0.97, fontweight="bold", fontname="serif"
        )
    plt.xlabel("Tiempo")
    ax1.set_ylabel("Precio")
    ax2.set_ylabel("Precio")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")


def plot_prices_tree(
    S: numpy.array, asset: str, option_type: str = None, params_dict: dict = {}
) -> None:
    """Grafica el árbol de precios del activo subyacente o de los payoffs de la
    opción.

    Parameters
    ----------
    S : numpy.array
        Arreglo que contiene los precios del activo subyacente o los payoffs de
        la opción, organizados en una matriz triangular superior, de dimensiones
        (n_paths, steps + 1), donde n_paths es el número de trayectorias y steps
        es el número de pasos en la discretización del tiempo.
    asset : str
        Indicador para graficar el árbol de precios del activo subyacente o de
        los payoffs de la opción. Puede ser 'subyacente' u 'opción'.
    option_type : str
        Tipo de opción ('call' o 'put').

    Returns
    -------
    None.
    """

    if asset == "subyacente":
        symb = "S"
        node_color = "tab:blue"
        node_alpha = 0.1
        option_type = ""
    elif asset == "opción":
        symb = "P"
        node_color = "#00d084"
        node_alpha = 0.4
        if option_type not in ["call", "put"]:
            raise ValueError("option_type debe ser 'call' o 'put'.")

    else:
        raise ValueError("asset debe ser 'subyacente' u 'opción'.")
    # Se extraen las dimensiones de la matriz de precios.
    S = S.round(2)
    n_paths = S.shape[0]
    steps = S.shape[1]

    # Lista de nodos para graficar el árbol de precios.
    edgelist = []

    # Se inicializa el diccionario que se utilizará para renombrar los nodos del
    # grafo con los precios.
    ind_to_labels = {}
    ind_to_cord = {}
    # Se intera sobre los rengloes de la matriz de precios.
    for i in range(n_paths - 1):
        # Se itera sobre las columnas de la matriz de precios.
        for j in range(steps - 1):
            # Condición para sólo recorrer las entradas de la matriz que están
            # en la diagonal superior.
            if j >= i:
                # Se agregan los vértices correspondientes al nodo S_u.
                edgelist.append(((i, j), (i, j + 1)))
                # Se agregan los vértices correspondientes al nodo S_d.
                edgelist.append(((i, j), (i + 1, j + 1)))
                # Mapeo de los índices a las etiquetas con los precios
                # en cada paso de tiempo.
                ind_to_labels[(i, j)] = (
                    f'$\\mathbf{{{ symb + "_{" + "u"*(j - i) + "d"*i + "}" }}}$'
                    f"\n\n ${S[i, j]}$"
                )

                # Se agrega la coordenada correspondiente al nodo (i, j).
                ind_to_cord[(i, j)] = (j, (j - i) - i)

    # Etiqueta del nodo inicial.
    ind_to_labels[(0, 0)] = f'$\\mathbf{{{symb + "_0"}}}$' f"\n\n ${S[0, 0]}$"
    # Coordenada del nodo inicial.
    ind_to_cord[(0, 0)] = (0, 0)

    for i in range(n_paths):
        # Coordenadas de los nodos finales.
        j = steps - 1
        ind_to_labels[(i, j)] = (
            f'$\\mathbf{{{ symb + "_{" + "u"*(j - i) + "d"*i + "}" }}}$'
            f"\n\n ${S[i, j]}$"
        )

        ind_to_cord[(i, j)] = (j, (j - i) - i)

    # Se crea el grafo con índices como nodos.
    ind_graph = networkx.Graph(edgelist)

    # Se etiquetan los nodos del grafo con los precios.
    labels_graph = networkx.relabel_nodes(ind_graph, ind_to_labels, copy=True)

    # A partir del diccionario 'ind_coord' se crea el diccionario 'labels_coord'
    # el cual se utilizará agregar las coordenadas de los nodos ya etiquetados.

    # label[indice] : coordenada[indice]
    labels_to_cord = {
        ind_to_labels[ind]: ind_to_cord[ind] for ind in ind_to_labels
        }

    # Estilo personalizado
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))

    # Se dibujan los nodos del grafo en las coordenadas correspondientes.
    options = {"edgecolors": "tab:gray", "node_size": 2000, "alpha": node_alpha}
    networkx.draw_networkx_nodes(
        labels_graph, labels_to_cord, node_color=node_color, **options
    )

    # Se dibujan los vértices del grafo, con forma de flecha, blanca.
    networkx.draw_networkx_edges(
        labels_graph,
        labels_to_cord,
        edge_color="whitesmoke",
        arrows=True,
        width=2,
        arrowsize=15,
        arrowstyle="->",
        min_target_margin=25,
        min_source_margin=30,
    )
    # Se agregan las etiquetas de los nodos.
    networkx.draw_networkx_labels(
        labels_graph, labels_to_cord, font_size=10, font_color="whitesmoke"
    )

    if asset == "subyacente":
        title = (
            f"Árbol de precios del activo subyacente ${symb}$ \n\n con: "
            ", ".join(
                [f"${k}$ = {round(v, 3)}" for k, v in params_dict.items()]
                )
        )
    elif asset == "opción":
        title = (
            f"Árbol de payoffs de la opción {option_type} ${symb}$ \n\n con: "
            + ", ".join(
                [f"${k}$ = {round(v, 3)}" for k, v in params_dict.items()]
                )
        )

    plt.title(title)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
