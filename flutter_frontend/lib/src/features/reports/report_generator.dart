import 'dart:typed_data';
import 'package:printing/printing.dart';
import 'pdf_report.dart';

Future<void> generateAndShareReport() async {
  final Uint8List pdfBytes = await buildPdfReport(
    title: 'Veilo Lung Cancer Report',
    summary: 'Summary placeholder',
  );
  await Printing.sharePdf(bytes: pdfBytes, filename: 'veilo_report.pdf');
}


