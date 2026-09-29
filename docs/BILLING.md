# Créditos, límites y costos

## Estado seguro

El backend opera en modo **cerrado por defecto**. Si no puede verificar identidad, saldo o límites en la base persistente, ningún endpoint de generación inicia una llamada al proveedor.

## Variables de entorno

Configurar únicamente en el entorno del backend, nunca en archivos del repositorio:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ALLOWED_ORIGINS` (orígenes separados por coma)

La clave de servicio nunca debe enviarse al navegador. La web utilizará solo la clave pública de Supabase para solicitar un código de verificación por correo o teléfono.

## Instalación

1. Crear un proyecto vacío en Supabase.
2. Ejecutar `supabase/migrations/001_billing.sql` en el editor SQL.
3. Configurar las tres variables anteriores en el backend.
4. Mantener desactivadas las llamadas de generación hasta que la web envíe una sesión verificada.
5. Ejecutar las pruebas y una generación controlada antes de desplegar.

## Reglas implementadas

- Cinco créditos gratuitos por identidad verificada, una sola vez.
- Los créditos gratuitos nunca se renuevan.
- Límites independientes para texto, imágenes, minutos de video, minutos de voz y canciones.
- Reserva atómica antes de usar cualquier motor.
- Reembolso automático de la reserva cuando la generación falla.
- Idempotencia para impedir cobros dobles por reintentos.
- Registro del motor, unidades, créditos y costo estimado de cada generación.
- El servidor no guarda prompts ni contenido generado en la base de facturación.

## Retención

- Archivos generados: objetivo de 7 días; la eliminación automática se implementará en la tarea de almacenamiento.
- Conversaciones: no se guardan en esta etapa.
- Eventos de créditos y costos: 24 meses para conciliación y análisis.
- Datos de cuenta: mientras exista la cuenta y hasta 30 días después de solicitar su eliminación, salvo obligaciones legales aplicables.

## Siguiente integración

La interfaz debe añadir verificación mínima por correo o teléfono y enviar el token de sesión en `Authorization: Bearer ...`. Hasta entonces el backend rechazará las generaciones y no causará gasto.
