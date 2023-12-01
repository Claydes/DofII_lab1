import model
import inference_mamdani
import fuzzy_operators


class FuzzyController:
    def __init__(self):
        inference_mamdani.preprocessing(model.input_lvs, model.output_lv)

    @staticmethod
    def normalization(x, x_min=1, x_max=10) -> float:
        return round((x - x_min) / (x_max - x_min), 2)

    @staticmethod
    def denormalization(y, y_min=1, y_max=10) -> float:
        return round(y * (y_max - y_min) + y_min, 1)

    def get_result(self, crisp):
        normalized_crisp = list(map(self.normalization, crisp[:3]))
        normalized_crisp.append(self.normalization(crisp[-1], 0, 20))
        result = inference_mamdani.process(model.input_lvs, model.output_lv, model.rule_base, normalized_crisp)
        for lv in model.input_lvs:
            fuzzy_operators.draw_lv(lv)
        fuzzy_operators.draw_lv(model.output_lv)
        return (result[1], self.denormalization(result[0]))



FuzzyController






