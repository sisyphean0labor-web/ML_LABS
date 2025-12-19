#ml
import numpy as np
from collections import Counter


def find_optimal_split(feature_values, target_labels):
    """Определяет наилучший порог разделения для признака"""
    feature_vector = np.asarray(feature_values)
    target_vector = np.asarray(target_labels)

    sorted_order = np.argsort(feature_vector)
    ordered_features = feature_vector[sorted_order]
    ordered_targets = target_vector[sorted_order]

    distinct_values = np.unique(ordered_features)

    if len(distinct_values) < 2:
        return np.array([]), np.array([]), None, None

    potential_thresholds = []
    for i in range(len(distinct_values) - 1):
        middle_point = (distinct_values[i] + distinct_values[i + 1]) / 2.0
        potential_thresholds.append(middle_point)

    potential_thresholds = np.array(potential_thresholds)

    total_instances = len(target_vector)
    positive_instances = np.sum(target_vector == 1)

    impurity_scores = []

    for threshold in potential_thresholds:
        left_condition = ordered_features < threshold
        right_condition = ~left_condition

        left_targets = ordered_targets[left_condition]
        right_targets = ordered_targets[right_condition]

        left_samples = len(left_targets)
        right_samples = len(right_targets)

        # Определяем количество положительных примеров в каждом подмножестве
        left_positives = np.sum(left_targets == 1)
        right_positives = np.sum(right_targets == 1)

        # Рассчитываем вероятности классов
        prob_left_pos = left_positives / left_samples if left_samples > 0 else 0
        prob_left_neg = 1 - prob_left_pos

        prob_right_pos = right_positives / right_samples if right_samples > 0 else 0
        prob_right_neg = 1 - prob_right_pos

        # Вычисляем индекс Джини для каждой части
        left_gini = 1.0 - (prob_left_pos ** 2 + prob_left_neg ** 2)
        right_gini = 1.0 - (prob_right_pos ** 2 + prob_right_neg ** 2)

        # Взвешенная мера неопределенности
        combined_gini = (left_samples / total_instances) * left_gini + \
                        (right_samples / total_instances) * right_gini

        impurity_scores.append(combined_gini)

    impurity_scores = np.array(impurity_scores)

    # Определяем индекс минимальной неопределенности
    optimal_index = np.argmin(impurity_scores)
    optimal_threshold = potential_thresholds[optimal_index]
    optimal_impurity = impurity_scores[optimal_index]

    return potential_thresholds, impurity_scores, optimal_threshold, optimal_impurity


class ClassificationTree:
    def __init__(self, feature_kinds, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        """Создает экземпляр дерева классификации"""
        permitted_kinds = {"real", "categorical"}
        for fkind in feature_kinds:
            if fkind not in permitted_kinds:
                raise ValueError("Недопустимый тип признака")

        self.root_node = {}
        self.feature_kinds = feature_kinds
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _build_node(self, data_matrix, label_vector, node_dict, depth_level=0):
        """Рекурсивно формирует узел дерева"""
        if self._halt_condition(label_vector, depth_level):
            self._create_leaf(node_dict, label_vector)
            return

        # Определяем оптимальное разделение среди всех признаков
        division_details = self._determine_best_division(data_matrix, label_vector)

        if division_details is None:
            self._create_leaf(node_dict, label_vector)
            return

        # Конфигурируем узел как внутренний
        self._setup_division_node(
            node_dict,
            division_details['feature_index'],
            division_details['division_point'],
            division_details['feature_kind']
        )

        # Разделяем данные на две группы
        left_selection, right_selection = self._partition_dataset(
            data_matrix[:, division_details['feature_index']],
            division_details['division_point'],
            division_details['feature_kind']
        )

        # Рекурсивно строим левое поддерево
        self._build_node(
            data_matrix[left_selection],
            label_vector[left_selection],
            node_dict['left_child'],
            depth_level + 1
        )

        # Рекурсивно строим правое поддерево
        self._build_node(
            data_matrix[right_selection],
            label_vector[right_selection],
            node_dict['right_child'],
            depth_level + 1
        )

    def _halt_condition(self, labels, current_depth):
        """Определяет, следует ли остановить построение дерева"""
        # Все метки идентичны - узел чистый
        distinct_labels = np.unique(labels)
        if len(distinct_labels) == 1:
            return True

        # Достигнут максимальный уровень глубины
        if self.max_depth is not None:
            if current_depth >= self.max_depth:
                return True

        # Недостаточно наблюдений для разделения
        if self.min_samples_split is not None:
            if len(labels) < self.min_samples_split:
                return True

        return False

    def _create_leaf(self, node, labels):
        """Формирует терминальный узел с предсказанием"""
        node['node_type'] = 'leaf'
        label_frequencies = Counter(labels)
        majority_class = label_frequencies.most_common(1)[0][0]
        node['output'] = majority_class

    def _determine_best_division(self, data, labels):
        """Анализирует все признаки и находит наилучшее разделение"""
        optimal_division = {
            'impurity': float('inf'),
            'feature_index': None,
            'division_point': None,
            'feature_kind': None
        }

        feature_count = data.shape[1]

        # Проверяем каждый признак
        for feat_idx in range(feature_count):
            feat_kind = self.feature_kinds[feat_idx]
            feature_col = data[:, feat_idx]

            # Обрабатываем в соответствии с типом признака
            if feat_kind == 'categorical':
                # Трансформируем категории в числовые значения
                transformed_data, category_dict = self._transform_categorical(feature_col, labels)
            elif feat_kind == 'real':
                # Для непрерывных признаков приводим к числовому типу
                transformed_data = feature_col.astype(float)
                category_dict = None
            else:
                raise ValueError(f"Неожиданный тип признака: {feat_kind}")

            # Определяем оптимальный порог для этого признака
            _, _, best_cut, best_impurity = find_optimal_split(transformed_data, labels)

            if best_impurity is None:
                continue

            # Проверяем минимальное количество образцов в листьях
            if not self._validate_leaf_requirements(transformed_data, best_cut):
                continue

            # Обновляем информацию о наилучшем разделении
            if best_impurity < optimal_division['impurity']:
                optimal_division['impurity'] = best_impurity
                optimal_division['feature_index'] = feat_idx
                optimal_division['division_point'] = self._adjust_division_point(
                    best_cut, feat_kind, category_dict
                )
                optimal_division['feature_kind'] = feat_kind

        if optimal_division['feature_index'] is None:
            return None

        return optimal_division

    def _transform_categorical(self, feature_data, labels):
        """Кодирует категориальный признак на основе целевых значений"""
        # Подсчитываем частоты категорий
        category_frequencies = Counter(feature_data)

        # Подсчитываем положительные метки для каждой категории
        positive_frequencies = Counter(feature_data[labels == 1])

        # Вычисляем отношение положительных примеров к общему количеству
        category_proportions = {}
        for category, total in category_frequencies.items():
            positive = positive_frequencies.get(category, 0)
            category_proportions[category] = positive / total

        # Упорядочиваем категории по возрастанию доли положительных
        sorted_categories = sorted(category_proportions.keys(),
                                   key=lambda cat: category_proportions[cat])

        # Создаем словарь для числового кодирования
        encoding_map = {cat: idx for idx, cat in enumerate(sorted_categories)}

        # Применяем кодирование к данным
        coded_values = np.array([encoding_map.get(val, 0) for val in feature_data])

        return coded_values, encoding_map

    def _validate_leaf_requirements(self, feature_data, division_point):
        """Проверяет соответствие размеров листьев минимальным требованиям"""
        if self.min_samples_leaf is None:
            return True

        # Определяем размеры левого и правого листьев
        left_leaf_size = np.sum(feature_data < division_point)
        right_leaf_size = len(feature_data) - left_leaf_size

        return (left_leaf_size >= self.min_samples_leaf and 
                right_leaf_size >= self.min_samples_leaf)

    def _adjust_division_point(self, division_point, feature_kind, category_data):
        """Адаптирует точку разделения в зависимости от типа признака"""
        if feature_kind == 'real':
            return division_point

        # Для категориальных признаков формируем список категорий для левого поддерева
        left_group = [cat for cat, code in category_data.items()
                      if code < division_point]
        return left_group

    def _setup_division_node(self, node, feature_idx, division_point, feature_kind):
        """Настраивает узел с разделением"""
        node['node_type'] = 'branch'
        node['selected_feature'] = feature_idx
        node['left_child'] = {}
        node['right_child'] = {}

        if feature_kind == 'real':
            node['cutoff_value'] = division_point
        else:
            node['category_group'] = division_point

    def _partition_dataset(self, feature_data, division_point, feature_kind):
        """Разделяет выборку на две подвыборки"""
        if feature_kind == 'real':
            left_indicator = feature_data.astype(float) < division_point
        else:
            left_indicator = np.isin(feature_data, division_point)

        right_indicator = ~left_indicator

        return left_indicator, right_indicator

    def _process_instance(self, sample, node):
        """Рекурсивно проходит по дереву для классификации объекта"""
        if node["node_type"] == "leaf":
            return node["output"]

        # Извлекаем информацию о разделении
        feature_idx = node["selected_feature"]
        fkind = self.feature_kinds[feature_idx]

        if fkind == "real":
            test_condition = float(sample[feature_idx]) < node["cutoff_value"]
            if test_condition:
                return self._process_instance(sample, node["left_child"])
            else:
                return self._process_instance(sample, node["right_child"])

        elif fkind == "categorical":
            test_condition = sample[feature_idx] in node["category_group"]
            if test_condition:
                return self._process_instance(sample, node["left_child"])
            else:
                return self._process_instance(sample, node["right_child"])
        else:
            raise ValueError("Неподдерживаемый тип признака")

    def train(self, X, y):
        """Выполняет обучение дерева классификации"""
        self.root_node = {}
        self._build_node(X, y, self.root_node)

    def classify(self, X):
        """Выполняет классификацию набора объектов"""
        classifications = [self._process_instance(x, self.root_node) for x in X]
        return np.array(classifications)