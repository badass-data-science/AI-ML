from membership_function import StandardMBF_Z, StandardMBF_S, StandardMBF_Lambda
from linguistic_set import LinguisticSet
from linguistic_domain import LinguisticDomain
from fuzzy_value import FuzzyValue


def main():
    print()
    print("This program demonstrates Emily Williams' yet untitled")
    print("fuzzy logic toolkit.")
    print()
    print("It implements the crane example in Constantin von Altrock's book")
    print("'Fuzzy Logic & Neurofuzzy Applications Explained'")
    print()
    print("Distance sensor reads 12 yards.")
    print("Angle sensor reads 4 degrees.")

    # Distance domain
    distance_domain = LinguisticDomain("distance_domain")
    distance_domain.add_linguistic_set(LinguisticSet(StandardMBF_Z(-5, 0),          "too_far"))
    distance_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-5, 0, 5),  "zero"))
    distance_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(0, 5, 10),  "close"))
    distance_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(5, 10, 30), "medium"))
    distance_domain.add_linguistic_set(LinguisticSet(StandardMBF_S(10, 30),         "far"))

    # Angle domain
    angle_domain = LinguisticDomain("angle_domain")
    angle_domain.add_linguistic_set(LinguisticSet(StandardMBF_Z(-45, -5),           "neg_big"))
    angle_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-45, -5, 0),   "neg_small"))
    angle_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-5, 0, 5),     "zero"))
    angle_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(0, 5, 45),     "pos_small"))
    angle_domain.add_linguistic_set(LinguisticSet(StandardMBF_S(5, 45),             "pos_big"))

    # Power domain
    power_domain = LinguisticDomain("power_domain")
    power_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-30, -25, -8), "neg_high"))
    power_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-25, -8, 0),  "neg_medium"))
    power_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(-8, 0, 8),    "zero"))
    power_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(0, 8, 25),    "pos_medium"))
    power_domain.add_linguistic_set(LinguisticSet(StandardMBF_Lambda(8, 25, 20),   "pos"))

    # Fuzzify sensor readings
    distance = FuzzyValue(distance_domain)
    distance.set_crisp_value(12)

    angle = FuzzyValue(angle_domain)
    angle.set_crisp_value(4)

    # Fuzzy inference
    power = FuzzyValue(power_domain)
    power.OR_set_set_membership(distance.AND("medium", angle, "pos_small"), "pos_medium")
    power.OR_set_set_membership(distance.AND("medium", angle, "zero"),      "zero")
    power.OR_set_set_membership(distance.AND("far",    angle, "zero"),      "pos_medium")

    # Defuzzify
    power_setting = power.get_crisp_value()
    print(f"Set power to {power_setting} kW.")
    print()


if __name__ == "__main__":
    main()
