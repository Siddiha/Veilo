import 'dart:io';
import 'package:flutter/material.dart';

class ImagePreview extends StatelessWidget {
  final File file;
  const ImagePreview({super.key, required this.file});

  @override
  Widget build(BuildContext context) {
    return Image.file(file, fit: BoxFit.contain);
  }
}


