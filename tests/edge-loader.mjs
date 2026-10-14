export async function resolve(specifier, context, nextResolve) {
  if (specifier === "https://esm.sh/@supabase/supabase-js@2") return {
    url: "data:text/javascript,export function createClient(...args){return globalThis.__createClient(...args)}",
    shortCircuit: true,
  };
  return nextResolve(specifier, context);
}
