package fuzzy;

public class Main {

    public static void main(String[] args) {
        System.out.println();
        System.out.println("This program demonstrates Emily Williams' yet untitled");
        System.out.println("fuzzy logic toolkit.");
        System.out.println();
        System.out.println("It implements the crane example in Constantin von Altrock's book");
        System.out.println("'Fuzzy Logic & Neurofuzzy Applications Explained'");
        System.out.println();
        System.out.println("Distance sensor reads 12 yards.");
        System.out.println("Angle sensor reads 4 degrees.");

        // Distance domain
        LinguisticDomain distanceDomain = new LinguisticDomain("distance_domain");
        distanceDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Z(-5, 0),           "too_far"));
        distanceDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-5, 0, 5),   "zero"));
        distanceDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(0, 5, 10),   "close"));
        distanceDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(5, 10, 30),  "medium"));
        distanceDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_S(10, 30),          "far"));

        // Angle domain
        LinguisticDomain angleDomain = new LinguisticDomain("angle_domain");
        angleDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Z(-45, -5),           "neg_big"));
        angleDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-45, -5, 0),   "neg_small"));
        angleDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-5, 0, 5),     "zero"));
        angleDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(0, 5, 45),     "pos_small"));
        angleDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_S(5, 45),             "pos_big"));

        // Power domain
        LinguisticDomain powerDomain = new LinguisticDomain("power_domain");
        powerDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-30, -25, -8), "neg_high"));
        powerDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-25, -8, 0),  "neg_medium"));
        powerDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(-8, 0, 8),    "zero"));
        powerDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(0, 8, 25),    "pos_medium"));
        powerDomain.addLinguisticSet(new LinguisticSet(new StandardMBF_Lambda(8, 25, 20),   "pos"));

        // Fuzzify sensor readings
        FuzzyValue distance = new FuzzyValue(distanceDomain);
        distance.setCrispValue(12);

        FuzzyValue angle = new FuzzyValue(angleDomain);
        angle.setCrispValue(4);

        // Fuzzy inference
        FuzzyValue power = new FuzzyValue(powerDomain);
        power.OR_setSetMembership(distance.AND("medium", angle, "pos_small"), "pos_medium");
        power.OR_setSetMembership(distance.AND("medium", angle, "zero"),      "zero");
        power.OR_setSetMembership(distance.AND("far",    angle, "zero"),      "pos_medium");

        // Defuzzify
        double powerSetting = power.getCrispValue();
        System.out.println("Set power to " + powerSetting + " kW.");
        System.out.println();
    }
}
