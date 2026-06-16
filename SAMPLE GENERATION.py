import json
import random
import math

def generate_vliw_problem(num_instructions, filename="vliw_instance.json"):
    """
    Генерирует задачу планирования для VLIW процессора и сохраняет в JSON.
    
    Args:
        num_instructions (int): Количество инструкций (I).
        filename (str): Имя выходного файла.
    """
    
    # ==========================================
    # 1. Конфигурация Архитектуры (VLIW Model)
    # ==========================================
    
    # Множество функциональных юнитов F
    # ID юнитов: 0: ALU0, 1: ALU1, 2: FPU0, 3: MEM0, 4: BR0
    functional_units = [
        {"id": 0, "name": "ALU_0", "type": "INT"},   # Integer Arithmetic
        {"id": 1, "name": "ALU_1", "type": "INT"},   # Integer Arithmetic (параллельный)
        {"id": 2, "name": "FPU_0", "type": "FLOAT"}, # Floating Point
        {"id": 3, "name": "MEM_0", "type": "MEM"},   # Load/Store Unit
        {"id": 4, "name": "BR_0",  "type": "BRANCH"} # Branch Unit
    ]
    F_ids = [fu["id"] for fu in functional_units]
    
    # Множество ресурсов K (например, порты кэша данных)
    # W_k - лимит ресурса k в один такт
    resources = [
        {"id": 0, "name": "L1_CACHE_PORT", "capacity": 1} 
    ]
    K_ids = [res["id"] for res in resources]
    W_limits = {res["id"]: res["capacity"] for res in resources}

    # ==========================================
    # 2. Генерация Инструкций (I) и их параметров
    # ==========================================
    
    instructions = []
    
    # Вероятности появления типов инструкций в "типичной" программе
    # INT: 50%, MEM: 30%, FLOAT: 15%, BRANCH: 5%
    types = ["INT", "MEM", "FLOAT", "BRANCH"]
    weights = [0.50, 0.30, 0.15, 0.05]
    
    for i in range(num_instructions):
        instr_type = random.choices(types, weights=weights, k=1)[0]
        
        # Параметры по умолчанию
        lat = 1
        capable_fus = [] # F(i)
        w_usage = {0: 0} # w_ik (использование кэша)
        
        if instr_type == "INT":
            lat = 1
            capable_fus = [0, 1] # Может выполняться на ALU0 или ALU1
            
        elif instr_type == "FLOAT":
            lat = 3 # FPU операции обычно дольше (3-4 такта)
            capable_fus = [2]
            
        elif instr_type == "MEM":
            lat = 2 # Load operations have latency (предполагаем L1 hit)
            capable_fus = [3]
            w_usage[0] = 1 # Потребляет 1 слот порта памяти
            
        elif instr_type == "BRANCH":
            lat = 1
            capable_fus = [4]
        
        instructions.append({
            "id": i,
            "type": instr_type,
            "latency": lat,       # lat_i
            "F_i": capable_fus,   # F(i)
            "w_ik": w_usage       # w_ik
        })

    # ==========================================
    # 3. Генерация Графа Зависимостей E(G)
    # ==========================================
    
    edges = []
    
    # Создаем DAG (Directed Acyclic Graph).
    # i всегда зависит от j, где j < i, чтобы избежать циклов.
    # Плотность графа регулируется параметром density.
    
    # Уровень "локальности" зависимостей. Инструкции чаще зависят от недавних.
    lookback_window = 15 
    
    for i in range(1, num_instructions):
        # Количество зависимостей для текущей инструкции
        num_deps = random.choices([0, 1, 2, 3], weights=[0.1, 0.5, 0.3, 0.1])[0]
        
        # Выбираем потенциальных родителей из окна [i-lookback, i-1]
        start_node = max(0, i - lookback_window)
        candidates = list(range(start_node, i))
        
        if candidates and num_deps > 0:
            # Выбираем уникальных родителей
            parents = random.sample(candidates, k=min(num_deps, len(candidates)))
            for p in parents:
                edges.append([p, i]) # Ребро p -> i (S_i >= S_p + lat_p)

    # ==========================================
    # 4. Расчет временного горизонта T
    # ==========================================
    
    # Для ILP важно ограничить множество T, иначе матрица будет огромной.
    # T_min = длина критического пути.
    # T_max = последовательное выполнение всех инструкций.
    
    # Грубая оценка критического пути (Critical Path Method - Forward Pass)
    earliest_start = [0] * num_instructions
    for i in range(num_instructions):
        # Находим max(ES_parent + lat_parent) для всех родителей
        predecessors = [u for u, v in edges if v == i]
        max_prev_finish = 0
        for p in predecessors:
            finish_p = earliest_start[p] + instructions[p]["latency"]
            if finish_p > max_prev_finish:
                max_prev_finish = finish_p
        earliest_start[i] = max_prev_finish
    
    critical_path_len = max([es + instructions[i]["latency"] for i, es in enumerate(earliest_start)])
    
    # Добавляем "запас" (slack), так как ресурсные конфликты увеличат время выполнения
    # по сравнению с бесконечными ресурсами (критический путь).
    # Для VLIW с параллелизмом ~2-3, коэффициент 1.5 - 2.0 обычно безопасен.
    time_horizon = int(critical_path_len * 2.0) + 5
    
    # ==========================================
    # 5. Сохранение результатов
    # ==========================================
    
    data = {
        "metadata": {
            "description": "VLIW Instruction Scheduling Instance",
            "num_instructions": num_instructions,
            "time_horizon": time_horizon,
            "critical_path_lb": critical_path_len
        },
        "sets": {
            "F": F_ids,
            "K": K_ids,
            "T": list(range(time_horizon))
        },
        "parameters": {
            "W_k": W_limits
        },
        "instructions": instructions, # Содержит lat_i, F(i), w_ik
        "dependencies": edges         # E(G) как список пар [i, j]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Генерация завершена.")
    print(f"Инструкций: {num_instructions}")
    print(f"Ребер зависимостей: {len(edges)}")
    print(f"Горизонт времени T: {time_horizon} (LB по CP: {critical_path_len})")
    print(f"Файл сохранен как: {filename}")

# Пример использования
if __name__ == "__main__":
    # Сгенерировать задачу на 20 инструкций (для теста)
    generate_vliw_problem(num_instructions=150, filename="vliw_input.json")
    
    # Сгенерировать задачу побольше (50 инструкций)
    # generate_vliw_problem(num_instructions=50, filename="vliw_data_medium.json")
