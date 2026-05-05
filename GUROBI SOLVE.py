import json
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

def plot_vliw_schedule(json_filename, schedule, assignment, title="VLIW Schedule"):
    print("Построение Pipelined графика...")
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if "instructions" in data: instr_data = data["instructions"]
    else: instr_data = data["data"]["instructions"]
    
    instr_meta = {i["id"]: i for i in instr_data}
    latencies = {i["id"]: i["latency"] for i in instr_data}
    
    # Имена юнитов
    UNIT_NAMES = {
        0: "ALU_0", 1: "ALU_1",
        2: "FPU_0", 3: "MEM_0", 4: "BR_0"
    }
    
    colors = {
        "INT": "#AED581", "MEM": "#FFB74D", "FLOAT": "#4DD0E1", 
        "BRANCH": "#E57373", "UNK": "#E0E0E0"
    }

    # Расчет размеров
    max_t = 0
    for iid, start in schedule.items():
        dur = latencies.get(iid, 1)
        if start + dur > max_t: max_t = start + dur
    
    fig_width = max(12, max_t * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Сортируем инструкции, чтобы рисовать сначала длинные хвосты, потом основные блоки
    sorted_items = sorted(schedule.items(), key=lambda x: x[1])

    for iid, start in sorted_items:
        dur = latencies.get(iid, 1)
        unit = assignment[iid]
        
        info = instr_meta.get(iid, {})
        itype = info.get("type", "UNK")
        c = colors.get(itype, colors["UNK"])
        
        # --- 1. Рисуем "Хвост" (Latency) ---
        # Показывает реальную длительность вычисления.
        # Делаем его полупрозрачным (alpha=0.3), чтобы он не мешал читать соседей.
        if dur > 1:
            ax.barh(y=unit, width=dur-1, left=start+1, height=0.4, 
                    color=c, edgecolor='none', alpha=0.3, zorder=5)

        # --- 2. Рисуем "Голову" (Issue Slot) ---
        # Это сам факт запуска инструкции. Ширина всегда 1 такт (или 0.95 для зазора).
        # Текст пишем СЮДА.
        rect = ax.barh(y=unit, width=0.95, left=start, height=0.7, 
                       color=c, edgecolor='black', linewidth=0.8, 
                       align='center', zorder=10)
        
        # --- 3. Текст ---
        # Теперь центр всегда предсказуем: start + 0.5
        center_x = start + 0.475 # Половина от 0.95
        center_y = unit
        
        # Уменьшаем шрифт, если ID длинный
        label_str = str(iid)
        fsize = 9 if len(label_str) < 3 else 7
        
        ax.text(center_x, center_y, label_str, 
                ha='center', va='center', 
                fontsize=fsize, color='black', weight='bold', 
                zorder=20, clip_on=True)

    ax.set_xlabel("Time (Cycles)")
    ax.set_ylabel("Functional Unit")
    ax.set_title(f"{title} (Execution flow)", fontsize=14)
    
    ax.set_xlim(0, max_t)
    ax.set_xticks(np.arange(0, max_t + 1, 1))
    
    # Имена осей
    used_units = sorted(list(set(assignment.values())))
    y_labels = [UNIT_NAMES.get(u, f"Unit_{u}") for u in used_units]
    ax.set_yticks(used_units)
    ax.set_yticklabels(y_labels)
    
    ax.grid(True, axis='x', which='major', linestyle='--', alpha=0.5, color='gray', zorder=0)
    
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items() if k != "UNK"]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=len(colors))
    
    plt.tight_layout()
    plt.show()

def solve_vliw_with_gurobi(json_filename="vliw_instance.json"):
    print(f"--- Запуск Gurobi Solver для {json_filename} ---")
    
    # 1. Загрузка данных 
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"Файл {json_filename} не найден.")

    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- Парсинг данных ---
    instructions_data = data["instructions"]
    I = [i["id"] for i in instructions_data]
    
    # Множества
    F_ids = data["sets"]["F"] # Все юниты
    K_ids = [int(k) for k in data["sets"].get("K", [])] # Все ресурсы
    
    # Временной горизонт
    # Если в JSON есть T, берем его длину. Иначе вычисляем запас.
    if "T" in data["sets"] and len(data["sets"]["T"]) > 0:
        T_horizon = len(data["sets"]["T"])
        T = list(range(T_horizon))
    else:
        # Эвристика: сумма латентностей
        total_lat = sum(i["latency"] for i in instructions_data)
        T_horizon = total_lat
        T = list(range(T_horizon))

    # Параметры
    lat = {i["id"]: i["latency"] for i in instructions_data}
    
    # F(i) - совместимые юниты
    F_i = {i["id"]: set(i["F_i"]) for i in instructions_data}
    
    # w_ik - потребление ресурсов {inst_id: {res_id: amount}}
    w_ik = {}
    for instr in instructions_data:
        usage = {}
        raw_wik = instr.get("w_ik", instr.get("w_usage", {}))
        for k, v in raw_wik.items():
            usage[int(k)] = v
        w_ik[instr["id"]] = usage
        
    # W_k - лимиты ресурсов
    W_k = {int(k): v for k, v in data["parameters"].get("W_k", {}).items()}

    # Граф зависимостей E(G)
    dependencies = []
    for pair in data["dependencies"]:
        dependencies.append((pair[0], pair[1])) # (i, j) где j зависит от i

    print(f"Загружено: {len(I)} инструкций, Горизонт T={T_horizon}")

    # =================================================================
    # 2. Построение Модели Gurobi
    # =================================================================
    
    # Создаем модель
    m = gp.Model("VLIW_Scheduling")
    
    
    m.setParam('OutputFlag', 1)
    # Лимит времени (в секундах), чтобы не ждать вечность на больших задачах
    m.setParam('TimeLimit', 30000) 

    print("Создание переменных...")
    
    # x[i,t,f] - бинарная: инструкция i начинается в t на юните f
    # Создаем только допустимые переменные (где f in F(i)) для экономии памяти
    x = {}
    for i in I:
        for t in T:
            for f in F_i[i]:
                x[i, t, f] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}_{f}")
    
    # S[i] - время начала (integer)
    S = m.addVars(I, vtype=GRB.INTEGER, lb=0, name="S")
    
    # Cmax - makespan (переменная для минимизации)
    Cmax = m.addVar(vtype=GRB.INTEGER, lb=0, name="Cmax")

    print("Создание ограничений...")
    
    # (2) Каждая инструкция запускается ровно один раз
    for i in I:
        m.addConstr(
            gp.quicksum(x[i, t, f] for t in T for f in F_i[i]) == 1, 
            name=f"Assign_{i}"
        )

    # (3) Связь S_i и x_itf: S_i = sum(t * x_itf)
    for i in I:
        m.addConstr(
            S[i] == gp.quicksum(t * x[i, t, f] for t in T for f in F_i[i]), 
            name=f"StartTime_{i}"
        )

    # (5) Зависимости: S_j >= S_i + lat_i
    for parent, child in dependencies:
        m.addConstr(
            S[child] >= S[parent] + lat[parent], 
            name=f"Dep_{parent}_{child}"
        )

    # (6) Емкость функциональных юнитов: 
    # В любой момент t на юните f может быть не более 1 инструкции
    for t in T:
        for f in F_ids:
            # Суммируем x по всем инструкциям, которые МОГУТ выполняться на f
            relevant_insts = [i for i in I if f in F_i[i]]
            if relevant_insts:
                m.addConstr(
                    gp.quicksum(x[i, t, f] for i in relevant_insts) <= 1,
                    name=f"UnitCap_{f}_{t}"
                )

    # (4) Ресурсы памяти/шины (W_k):
    # В любой момент t сумма потребления w_ik <= W_k
    for t in T:
        for k in K_ids:
            limit = W_k.get(k, 9999) # Если лимита нет, считаем большим
            # Собираем выражение потребления
            expr = gp.LinExpr()
            for i in I:
                amount = w_ik[i].get(k, 0)
                if amount > 0:
                    # Суммируем по всем f, где i может запуститься
                    for f in F_i[i]:
                        expr += amount * x[i, t, f]
            
            m.addConstr(expr <= limit, name=f"ResCap_{k}_{t}")

    # (1) Целевая функция: Min(Cmax)
    # Cmax >= S_i + lat_i для всех i
    for i in I:
        m.addConstr(Cmax >= S[i] + lat[i], name=f"DefCmax_{i}")
        
    m.setObjective(Cmax, GRB.MINIMIZE)

    # =================================================================
    # 3. Решение и Вывод
    # =================================================================
    
    print("Запуск оптимизации...")
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        print("\n" + "="*40)
        print(f"Статус решения: {m.status}")
        print(f"Минимальный Makespan (такты): {int(Cmax.X)}")
        print("="*40)
        
        # Сбор и валидация расписания для вывода
        final_schedule = {}
        final_assignment = {}
        print(f"{'Instr':<6} | {'Start':<6} | {'End':<6} | {'Unit':<10} | {'Type'}")
        print("-" * 50)
        
        sorted_I = sorted(I, key=lambda idx: S[idx].X)
        
        for i in sorted_I:
            start_time = int(S[i].X)
            final_schedule[i] = start_time
            end_time = start_time + lat[i]
            
            # Находим юнит
            used_unit = "Unknown"
            for t in T:
                for f in F_i[i]:
                    if x[i, t, f].X > 0.5: # Если переменная равна 1
                        used_unit = f
                        final_assignment[i] = f
                        break
            
            # Находим тип для красоты (из исходных данных)
            itype = "UNK"
            for data_i in instructions_data:
                if data_i["id"] == i:
                    itype = data_i.get("type", "UNK")
                    break
                    
            #print(f"{i:<6} | {start_time:<6} | {end_time:<6} | {used_unit:<10} | {itype}")
                
        plot_vliw_schedule(json_filename, final_schedule, final_assignment, title="VLIW Schedule")
    elif m.status == GRB.INFEASIBLE:
        print("Модель неразрешима!")
        # m.computeIIS() # Можно включить для отладки, если есть лицензия
    else:
        print(f"Решение не найдено. Статус: {m.status}")

if __name__ == "__main__":
    try:
        solve_vliw_with_gurobi("vliw_input.json")
    except gp.GurobiError as e:
        print(f"Ошибка Gurobi: {e}")
    except Exception as e:
        print(f"Ошибка: {e}")
