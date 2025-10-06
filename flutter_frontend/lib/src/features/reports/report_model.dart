class ReportModel {
  final double probability;
  final String riskLevel;
  final Map<String, dynamic> questionnaire;
  final List<Map<String, dynamic>> annotations;

  ReportModel({
    required this.probability,
    required this.riskLevel,
    required this.questionnaire,
    required this.annotations,
  });
}


