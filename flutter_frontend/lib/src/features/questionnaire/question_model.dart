class QuestionnaireModel {
  int age;
  String gender;
  int smokingYears;
  double packsPerDay;
  List<String> symptoms;
  bool familyHistory;

  QuestionnaireModel({
    required this.age,
    required this.gender,
    required this.smokingYears,
    required this.packsPerDay,
    required this.symptoms,
    required this.familyHistory,
  });
}


