import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt

# Sympy ile LaTeX parse
import sympy as sp
from sympy.parsing.latex import parse_latex

# PyMOO tek amaçlı fonksiyonlar
from pymoo.problems.single import rosenbrock, rastrigin, ackley, sphere
try:
    from pymoo.problems.single import zakharov, schwefel, griewank
    HAS_ZAKHAROV = True
    HAS_SCHWEFEL = True
    HAS_GRIEWANK = True
except ImportError:
    HAS_ZAKHAROV = False
    HAS_SCHWEFEL = False
    HAS_GRIEWANK = False

# Projedeki algoritmalar
from algorithms.genetic_algorithm import genetic_algorithm
from algorithms.big_bang_big_crunch import big_bang_big_crunch
from algorithms.grey_wolf_optimizer import grey_wolf_optimizer
from algorithms.harris_hawks_optimization import harris_hawks_optimization
from algorithms.bat_algorithm import bat_algorithm
from algorithms.pso_algorithm import pso  

# ---------------------------------------------------------
# 1) PyMOO Hazır Fonksiyon Sözlüğü
# ---------------------------------------------------------
pymoo_benchmarks = {
    "Rosenbrock": rosenbrock.Rosenbrock,
    "Rastrigin": rastrigin.Rastrigin,
    "Ackley": ackley.Ackley,
    "Sphere": sphere.Sphere
}
if HAS_ZAKHAROV:
    pymoo_benchmarks["Zakharov"] = zakharov.Zakharov
if HAS_SCHWEFEL:
    pymoo_benchmarks["Schwefel"] = schwefel.Schwefel
if HAS_GRIEWANK:
    pymoo_benchmarks["Griewank"] = griewank.Griewank

def wrap_pymoo(problem_class, n_var):
    """PyMOO single-objective problem -> 'func(*vars)->float' formunda sarmalayıcı."""
    problem = problem_class(n_var=n_var)

    def f(*vars):
        x = np.array(vars, dtype=float)
        val = problem.evaluate(x)
        # PyMOO evaluate -> array, mesela [skor]
        # Komplekse düşmemeli ama yine de check edilebilir
        if isinstance(val, (list, np.ndarray)):
            val = val[0]
        if isinstance(val, complex):
            # Sonsuz gibi davran
            return float('inf')
        return float(val)  # reel'e çevir
    return f

# ---------------------------------------------------------
# 2) LaTeX Parse Yardımcı Fonksiyonlar
# ---------------------------------------------------------
i = sp.Symbol("i", real=True)
n = sp.Symbol("n", real=True)
x_i = sp.Symbol("x_i", real=True)

locals_dict = {
    "i": i,
    "n": n,
    "x_i": x_i,
    "Sum": sp.Sum,
    "Product": sp.Product,
    "Abs": sp.Abs,
    "Eq": sp.Eq,
    "sqrt": sp.sqrt,
    "cos": sp.cos,
    "sin": sp.sin,
    "log": sp.log,
    "exp": sp.exp
}

def fix_subscripts(expr_str: str) -> str:
    replacements = {
        "x_{i}": "x_i"
    }
    out = expr_str
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out

def parse_and_fix_latex(latex_expression: str) -> str:
    parsed_expr = parse_latex(latex_expression)
    parsed_str = str(parsed_expr)
    fixed_str = fix_subscripts(parsed_str)
    final_expr = sp.sympify(fixed_str, locals=locals_dict)
    return str(final_expr)

def parse_and_display_latex_raw(latex_expression: str):
    parsed_expr = parse_latex(latex_expression)
    st.text(f"Original LaTeX: {latex_expression}")
    st.text(f"Parsed Expression (raw): {parsed_expr}")
    return str(parsed_expr)


# ---------------------------------------------------------
# 3) Streamlit Arayüz
# ---------------------------------------------------------
st.title("Heuristic Algorithms Solver (PyMOO + LaTeX)")

# -------- Fonksiyon Girişi --------
st.subheader("1. Matematiksel Fonksiyon Girişi")

input_mode = st.radio(
    "Fonksiyon giriş modu:",
    ["Metin", "LaTeX", "Hazır Fonksiyonlar (PyMOO)", "Geçmişten Seç"]
)

if "history" not in st.session_state:
    st.session_state["history"] = []

function_input = ""
selected_benchmark = None

if input_mode == "Metin":
    function_input = st.text_input("Fonksiyon girin (örn: x**2 + y**2).")

elif input_mode == "LaTeX":
    latex_input = st.text_input(
        r"LaTeX formatında fonksiyon girin (örn: \log(x + 1) + x^2 \cdot e^y - y \cdot \sqrt{x})"
    )
    if latex_input:
        try:
            _ = parse_and_display_latex_raw(latex_input)
            fixed_expr_str = parse_and_fix_latex(latex_input)
            function_input = fixed_expr_str
            st.text(f"Final Expression (with fix): {fixed_expr_str}")
        except Exception as e:
            st.error(f"LaTeX fonksiyonu işlenemedi: {e}")

elif input_mode == "Hazır Fonksiyonlar (PyMOO)":
    selected_benchmark = st.selectbox("PyMOO Fonksiyonu Seçin:", list(pymoo_benchmarks.keys()))
    if selected_benchmark:
        st.info(f"PyMOO fonksiyonu seçildi: {selected_benchmark}")
        function_input = f"PyMOO::{selected_benchmark}"

elif input_mode == "Geçmişten Seç":
    if st.session_state["history"]:
        function_input = st.selectbox("Geçmiş fonksiyonlar:", st.session_state["history"])
    else:
        st.warning("Henüz geçmiş fonksiyon yok.")


# Geçmişe ekleme
if function_input and function_input not in st.session_state["history"]:
    st.session_state["history"].append(function_input)

# Sympy parse (Metin / LaTeX / Geçmiş)
sympy_expr = None
if input_mode in ["Metin", "LaTeX", "Geçmişten Seç"]:
    if function_input.startswith("PyMOO::"):
        pass
    elif function_input:
        try:
            sympy_expr = sp.sympify(function_input, locals=locals_dict)
        except Exception as e:
            st.error(f"Sympy parse hatası: {e}")
            sympy_expr = None


# -------- Ortak Parametreler --------
st.subheader("2. Ortak Parametreler")

dimension = st.number_input("Dimension (n):", min_value=1, value=5, step=1)
population_size = st.number_input("Popülasyon Boyutu:", min_value=1, value=30, step=1)
iterations = st.number_input("İterasyon (Döngü) Sayısı:", min_value=1, value=50, step=1)

st.markdown("Her **değişken** için aynı alt-üst sınırı kullanacağız.")
bounds_str = st.text_input("Alt ve üst sınırları girin (örn: -10,10)", value="-10,10")

def parse_bounds(bounds_input):
    try:
        parts = bounds_input.split(",")
        lower = float(parts[0])
        upper = float(parts[1])
        return (lower, upper)
    except:
        return (-10, 10)

bounds_tuple = parse_bounds(bounds_str)

# -------- Algoritma Seçimi --------
algorithms = [
    "Genetik Algoritma",
    "Büyük Patlama Algoritması",
    "Gri Kurt Optimizasyonu",
    "Harris Hawks Optimization",
    "Bat Algorithm",
    "PSO"
]
st.subheader("3. Algoritma Seçimi ve Özel Parametreler")

selected_algorithm = st.selectbox("Kullanmak istediğiniz algoritmayı seçin:", algorithms)
algo_params = {}

if selected_algorithm == "Genetik Algoritma":
    with st.expander("Genetik Algoritma Parametreleri", expanded=True):
        crossover_rate = st.slider("Crossover Oranı", 0.0, 1.0, 0.8, 0.05)
        mutation_rate = st.slider("Mutasyon Oranı", 0.0, 1.0, 0.1, 0.01)
    algo_params["crossover_rate"] = crossover_rate
    algo_params["mutation_rate"] = mutation_rate

elif selected_algorithm == "Büyük Patlama Algoritması":
    with st.expander("Büyük Patlama Parametreleri", expanded=True):
        shrink_factor = st.slider("Daralma Katsayısı (0.0 - 1.0)", 0.0, 1.0, 0.5, 0.05)
    algo_params["shrink_factor"] = shrink_factor

elif selected_algorithm == "Gri Kurt Optimizasyonu":
    with st.expander("Gri Kurt Parametreleri", expanded=True):
        a_start = st.slider("a Başlangıç (2.0 önerilir)", 0.0, 5.0, 2.0, 0.1)
        a_end = st.slider("a Bitiş (0.0 önerilir)", 0.0, 5.0, 0.0, 0.1)
    algo_params["a_start"] = a_start
    algo_params["a_end"] = a_end

elif selected_algorithm == "Harris Hawks Optimization":
    with st.expander("HHO Parametreleri", expanded=True):
        e0 = st.slider("Kaçış Enerjisi Başlangıç (E0)", 0.0, 2.0, 1.0, 0.1)
    algo_params["e0"] = e0

elif selected_algorithm == "Bat Algorithm":
    with st.expander("Bat Algorithm Parametreleri", expanded=True):
        freq_min = st.slider("Frekans Min", 0.0, 2.0, 0.0, 0.1)
        freq_max = st.slider("Frekans Max", 0.0, 10.0, 2.0, 0.5)
        alpha_bat = st.slider("Alpha (zayıflama faktörü)", 0.0, 1.0, 0.9, 0.05)
        gamma_bat = st.slider("Gamma (nabız artışı)", 0.0, 1.0, 0.9, 0.05)
    algo_params["freq_min"] = freq_min
    algo_params["freq_max"] = freq_max
    algo_params["alpha_bat"] = alpha_bat
    algo_params["gamma_bat"] = gamma_bat

elif selected_algorithm == "PSO":
    with st.expander("PSO Parametreleri", expanded=True):
        w_max = st.slider("w Max (Atalet katsayısı başlangıç)", 0.0, 1.5, 0.9, 0.05)
        w_min = st.slider("w Min (Atalet katsayısı bitiş)", 0.0, 1.5, 0.2, 0.05)
        c1 = st.slider("Bireysel öğrenme katsayısı (c1)", 0.0, 5.0, 2.0, 0.1)
        c2 = st.slider("Sosyal öğrenme katsayısı (c2)", 0.0, 5.0, 2.0, 0.1)
        v_max = st.slider("Maksimum hız (vMax)", 0.0, 10.0, 6.0, 0.5)
    algo_params["w_max"] = w_max
    algo_params["w_min"] = w_min
    algo_params["c1"] = c1
    algo_params["c2"] = c2
    algo_params["v_max"] = v_max


# -------- Algoritmayı Çalıştırma --------
st.subheader("4. Çalıştırma ve Sonuçlar")

if st.button("Algoritmayı Çalıştır"):
    func = None

    # 1) PyMOO fonksiyonu
    if function_input.startswith("PyMOO::"):
        bench_name = function_input.replace("PyMOO::", "")
        if bench_name in pymoo_benchmarks:
            problem_class = pymoo_benchmarks[bench_name]
            func = wrap_pymoo(problem_class, dimension)
        else:
            st.error(f"PyMOO içinde '{bench_name}' tanımlı değil.")
            func = None

    # 2) Sympy'den gelen ifade (LaTeX, Metin veya Geçmiş)
    elif sympy_expr is not None:
        free_syms = sorted(sympy_expr.free_symbols, key=lambda s: s.name)
        num_free = len(free_syms)
        if num_free == 0:
            st.error("Girilen fonksiyonda değişken bulunamadı (sabit fonksiyon)!")
        else:
            # dimension != num_free ise override
            if num_free != dimension:
                dimension = num_free
                st.info(f"Girilen fonksiyonda {num_free} adet değişken tespit edildi. Dimension = {dimension} olarak ayarlandı.")

            # Lambdify
            try:
                raw_func = sp.lambdify(free_syms, sympy_expr, "numpy")

                # Burada "kompleks check" ekliyoruz
                def func(*vars):
                    val = raw_func(*vars)
                    # Eğer complex ise => inf
                    if isinstance(val, complex):
                        return float('inf')
                    # Eğer array/dizi dönmesi durumu (bir şekilde), 1. elemanı al vs.
                    if isinstance(val, (list, np.ndarray)):
                        if len(val) == 0:
                            return float('inf')
                        val = val[0]  # varsayılan
                        if isinstance(val, complex):
                            return float('inf')
                    # sqrt, log, vb. negatif domain => complex => inf
                    return float(val)

            except Exception as e:
                st.error(f"Lambdify hatası: {e}")
                func = None
    else:
        st.error("Geçerli bir fonksiyon seçilmedi veya parse edilemedi.")
        func = None

    # -----------------------------------------
    # Algoritma Koşumu
    # -----------------------------------------
    if func is not None:
        lower, upper = bounds_tuple
        num_variables = dimension

        try:
            if selected_algorithm == "Genetik Algoritma":
                best_sol, best_val, history = genetic_algorithm(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    crossover_rate=algo_params.get("crossover_rate", 0.8),
                    mutation_rate=algo_params.get("mutation_rate", 0.1)
                )
            elif selected_algorithm == "Büyük Patlama Algoritması":
                best_sol, best_val, history = big_bang_big_crunch(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    shrink_factor=algo_params.get("shrink_factor", 0.5)
                )
            elif selected_algorithm == "Gri Kurt Optimizasyonu":
                best_sol, best_val, history = grey_wolf_optimizer(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    a_start=algo_params.get("a_start", 2.0),
                    a_end=algo_params.get("a_end", 0.0)
                )
            elif selected_algorithm == "Harris Hawks Optimization":
                best_sol, best_val, history = harris_hawks_optimization(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    e0=algo_params.get("e0", 1.0)
                )
            elif selected_algorithm == "Bat Algorithm":
                best_sol, best_val, history = bat_algorithm(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    freq_min=algo_params.get("freq_min", 0.0),
                    freq_max=algo_params.get("freq_max", 2.0),
                    alpha_bat=algo_params.get("alpha_bat", 0.9),
                    gamma_bat=algo_params.get("gamma_bat", 0.9)
                )
            elif selected_algorithm == "PSO":
                best_sol, best_val, history = pso(
                    func,
                    num_variables,
                    (lower, upper),
                    pop_size=population_size,
                    max_iter=iterations,
                    w_max=algo_params.get("w_max", 0.9),
                    w_min=algo_params.get("w_min", 0.2),
                    c1=algo_params.get("c1", 2.0),
                    c2=algo_params.get("c2", 2.0),
                    v_max=algo_params.get("v_max", 6.0)
                )
            else:
                st.error("Algoritma tanınmıyor veya henüz implement edilmedi.")
                best_sol, best_val, history = None, None, None

            if best_sol is not None:
                st.success(f"En iyi çözüm (yaklaşık): {best_sol}")
                st.success(f"Fonksiyonun en iyi değeri: {best_val}")
                fig, ax = plt.subplots()
                ax.plot(range(1, len(history)+1), history, marker='o')
                ax.set_xlabel("İterasyon")
                ax.set_ylabel("En İyi Değer (Fitness)")
                ax.set_title(f"{selected_algorithm} - İterasyon Bazlı Değer")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
