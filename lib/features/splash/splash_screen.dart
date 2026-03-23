import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../core/widgets/primary_button.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'SignBridge',
              style: TextStyle(fontSize: 36, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 48),
            PrimaryButton(
              label: 'Iniciar sesión',
              onPressed: () => Navigator.pushNamed(context, '/login'),
            ),
            const SizedBox(height: 16),
            PrimaryButton(
              label: 'Registrarse',
              onPressed: () => Navigator.pushNamed(context, '/register'),
            ),
          ],
        ),
      ),
    );
  }
}
