import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../core/widgets/primary_button.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Container(
          width: double.infinity,
          height: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center, // Centrado vertical
            crossAxisAlignment: CrossAxisAlignment.center, // Centrado horizontal
            mainAxisSize: MainAxisSize.min, // Ajusta la columna al contenido
            children: [
              // Logo
              Image.asset(
                'assets/images/logov2.png',
                width: 200, // Ajusta según tu diseño
                height: 200,
              ),
              const SizedBox(height: 24),

              // Título
              const Text(
                'SignBridge',
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),

              // Botones
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
      ),
    );
  }
}