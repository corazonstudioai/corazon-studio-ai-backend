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

## Reglas de trabajo vigentes (13 de octubre de 2026)

- Nunca acceder al correo, cuentas personales o teléfono de la usuaria. Si un
  proceso exige verificación, pedirle que la haga ella, explicar el motivo y
  esperar su confirmación. No buscar ni solicitar códigos como si se tuviera
  acceso a esos medios.
- Pedir autorización para activar/pagar servicios nuevos, operar fuera de GitHub
  y Supabase, o integrar cambios grandes a producción.
- Continuar autónomamente con auditorías, correcciones y pruebas internas.
- Trabajar por bloques completos. Auditar dependencias, corregir causas raíz y
  no declarar un bloque terminado por una prueba superficial.
- Al cerrar una sesión, reportar solo avance verificable del plan total,
  funciones completamente terminadas/probadas y decisiones necesarias.

## Bloque Brain bajo revisión

La rama propone planificación estructurada de lenguaje natural, seguida de una
ejecución acotada de texto, imagen, voz o video. Los pedidos de funcionalidades
todavía no integradas reciben una explicación; no se ignoran silenciosamente.
El texto/plan consume un crédito. Una creación de archivos posterior consume
otro: se muestra este costo en la interfaz. Un fallo de creación devuelve el
crédito de esa creación, no el de un plan entregado satisfactoriamente.

Dependencias auditadas: reservas, límites, identidad, cache de resultados,
almacenamiento privado, firma de enlaces, generación de audio y seguimiento de
video. La migración 004 serializa reservas, vincula cada clave a su solicitud,
guarda resultados durables y liquida estado/créditos del video en una transacción.
La interfaz conserva los identificadores de las etapas para recuperar solicitudes
sin volver a planificar ni volver a generar archivos que ya existen.

Aceptación aislada: contratos del plan, integración de los handlers Edge con
proveedores simulados, fallos de almacenamiento, ausencia de sesión, límites,
repeticiones, recuperación y concurrencia. PostgreSQL se prueba exclusivamente
en un contenedor efímero del workflow de PR, no en la base de la usuaria.

No se activaron cuentas, facturación ni servicios externos; no se verificaron
correos reales; no se consumieron proveedores de IA reales. Esta rama requiere
aprobación antes de integrarse y antes de cualquier validación con proveedores
externos. Las pruebas internas no certifican la calidad de la respuesta del
modelo ni un archivo real descargado y reproducido.

## Medición del plan total

No hay todavía una ponderación acordada del plan ni una batería completa de
aceptación real de todas sus funciones. No usar conteos de archivos, pruebas o
commits como porcentaje de producto terminado. Mantener el porcentaje total como
no verificable hasta establecer y ejecutar una matriz integral. Funciones aún
sin aceptación real: Brain, identidad/interfaz, créditos, conversación, imágenes,
voz/dictado, video, música, subtítulos, traducción, eliminación de fondo y
exportación. No avanzar a otro bloque mientras Brain no cierre su aceptación.

## Evidencia de esta sesión

- Los 29 escenarios de integración local usan los handlers reales y proveedores
  simulados; no gastan saldo de IA.
- El workflow de aceptación de Brain aprobó en PostgreSQL 17 aislado: reaplicación
  de migraciones, primer uso y cuotas, cache, permisos, devolución idempotente,
  cierre transaccional del video y ocho solicitudes simultáneas con un solo cargo.
- Los controles Deno/tipos y las pruebas existentes del backend aprobaron en el PR.
- El workflow de navegador móvil aprobó el recorrido real de interfaz con todas
  las solicitudes interceptadas: cancelación, código ficticio de ocho dígitos con
  espacios, texto, carga de audio/imagen y recuperación sin repetir el plan.
- No se realizaron integración a main, despliegue, pruebas con cuentas personales
  ni generación con proveedores reales. Brain sigue pendiente de esa aceptación.
