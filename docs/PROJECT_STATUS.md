# Corazón Studio IA: continuidad del plan

## Decisión de infraestructura (13 de octubre de 2026)

Continuar con Supabase. No migrar a Neon ni eliminar servicios o datos por la
pausa de inactividad del plan gratuito. La comparación queda como alternativa
futura si los límites afectan el uso real. No se ha autorizado crear una cuenta
en Neon, transferir datos ni habilitar facturación.

Neon se evaluó por su base PostgreSQL con reactivación automática y sus servicios
de autenticación, archivos y funciones. Si se vuelve a considerar, verificar las
condiciones vigentes en https://neon.com/pricing y probar la compatibilidad antes
de decidir. La infraestructura gratuita no cubre el consumo de proveedores de IA.

## Alcance original

Completar Corazón Brain, interfaz sencilla ES/EN, texto, imágenes, voz, video,
música, subtítulos, traducción, eliminación de fondo y descarga de resultados.
El objetivo es resolver las necesidades de la usuaria; ninguna función se congela
por principio. Trabajar dentro de los repositorios backend y web. No acceder a
cuentas externas, secretos ni la página publicada sin autorización separada.

## Incremento actual

- Primera capa de enrutamiento de Brain en la interfaz: distinguir un guion de
  la generación de video; respetar el modo elegido explícitamente. Sigue siendo
  un enrutador basado en reglas, no el orquestador completo de lenguaje natural.
- Voz: separar texto hablado e instrucciones de estilo; rechazar textos de más
  de 1200 caracteres antes de reservar créditos, en lugar de truncarlos.
- Chat: aplicar el idioma de salida seleccionado y rechazar respuestas vacías
  antes de marcar la generación como exitosa.
- Interfaz: cerrar correctamente la autenticación cancelada; mostrar errores
  de créditos/sesión como tales. Consultar al servidor por videos pendientes,
  sin afirmar una devolución basada únicamente en el tiempo del navegador.

## Validación pendiente

El código y las pruebas no equivalen a un despliegue ni a una generación real.
Reactivar Supabase manualmente y comprobar registro, chat ES/EN, voz, video con
audio, recuperación tras recarga y crédito final. No se consumieron servicios
de IA reales durante estas correcciones. El orquestador completo, dictado,
subtítulos, música y exportación de formatos requieren implementación/validación
adicional; este incremento no los declara terminados.
