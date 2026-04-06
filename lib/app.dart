import 'package:flutter/material.dart';
import 'package:signbridge/features/saved_messages/saved_list_screen.dart';
import 'package:signbridge/features/translate/translate_screen.dart';
import 'core/theme/app_theme.dart';
import 'features/splash/splash_screen.dart';
import 'features/auth/login_screen.dart';
import 'features/auth/register_screen.dart';
import 'features/home/home_screen.dart';
import 'features/record/record_screen.dart';

class SignBridgeApp extends StatelessWidget {
  const SignBridgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignBridge',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      initialRoute: '/',
      routes: {
        '/': (context) => const SplashScreen(),
        '/login': (context) => const LoginScreen(),
        '/register': (context) => const RegisterScreen(),
        '/home': (context) => const HomeScreen(),
        '/record': (context) => RecordScreen(),
        '/translate': (context) => const TranslateScreen(),
        '/saved': (context) => const SavedListScreen(),
      },
    );
  }
}