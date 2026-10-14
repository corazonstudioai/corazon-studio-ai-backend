import assert from "node:assert/strict";
import { register } from "node:module";
register("./edge-loader.mjs", import.meta.url);
const handlers = {};
const configuration = { SUPABASE_URL:"https://test.supabase.co", SUPABASE_ANON_KEY:"test-anon", SUPABASE_SERVICE_ROLE_KEY:"test-service", OPENAI_API_KEY:"test-provider", FAL_KEY:"test-fal" };
globalThis.Deno = {env:{get:key=>configuration[key]},serve:handler=>{globalThis.latestHandler=handler;}};
let videoJob=null,mergeClaim=true,videoQueryError=false,settlementError=false,queueCount=0;
let providerCalls=0, calls=[], replay=null, receiptReason="reserved", reserveAllowed=true, signedFailure=false, storageFailure=false, saveFailure=false, providerOk=true, responseValue;
const service={
  rpc:async(name,args)=>{
    calls.push({name,args});
    if(name==="reserve_generation_v2")return{data:[{allowed:reserveAllowed,reason:reserveAllowed?receiptReason:"credits_exhausted",reservation_id:"test-reservation",credits_remaining:4}],error:null};
    if(name==="settle_video_job")return settlementError?{error:{message:"failed"}}:{data:{status:args.p_success?"completed":"failed",video_url:args.p_video_url,refunded:!args.p_success},error:null};
    if(name==="save_generation_result")return{error:saveFailure?{message:"failed"}:null};
    return{error:null};
  },
  from:table=>{
    const query={operation:null,filters:{},select(){return this},eq(key,value){this.filters[key]=value;return this},is(){return this},in(){return this},update(value){this.operation=value;return this},insert(value){this.inserted=value;return this},
      async maybeSingle(){
        if(table!=="video_jobs")return{data:replay?{result:replay}:null,error:null};
        if(this.filters.user_id&&this.filters.user_id!==videoJob?.user_id)return{data:null,error:null};
        return{data:videoJob,error:videoQueryError?{}:null};
      },
      async single(){return{data:{id:"created-video",status:"queued"},error:null}},
      async result(){
        if(this.operation?.merge_claimed_at){const granted=mergeClaim;mergeClaim=false;return{data:granted?[{id:"video-job"}]:[],error:null};}
        return{data:null,error:null};
      },
      then(resolve,reject){return this.result().then(resolve,reject)}
    };return query;
  },
  storage:{from:()=>({upload:async()=>({error:storageFailure?{}:null}),createSignedUrl:async()=>({data:{signedUrl:"https://test.supabase.co/generated/result"},error:signedFailure?{}:null})})},
};
let verified=true,authenticated=true;
globalThis.__createClient=(_url,key)=>key==="test-service"?service:{auth:{getUser:async()=>({data:{user:authenticated?{id:"test-user",email_confirmed_at:verified?"2026-01-01":null}:null},error:null})}};
globalThis.fetch=async(url,options)=>{
  if(String(url).startsWith("https://queue.fal.run/")){
    if(options?.method==="POST"){calls.push({name:"fal",url:String(url),body:JSON.parse(options.body)});queueCount++;return Response.json({request_id:"queued",status_url:"https://queue.fal.run/status",response_url:"https://queue.fal.run/result"});}
    if(String(url).endsWith("/result"))return videoJob?.merge_request_id?Response.json({video_url:"https://example.invalid/video.mp4"}):Response.json({video:{url:"https://example.invalid/video.mp4"}});
    return Response.json({status:"COMPLETED"});
  }
  assert.ok(String(url).startsWith("https://api.openai.com/v1/"),"Tests cannot reach arbitrary services");
  providerCalls++;calls.push({name:"provider",body:JSON.parse(options.body)});
  return responseValue instanceof Uint8Array?new Response(responseValue,{status:providerOk?200:500}):Response.json(responseValue,{status:providerOk?200:500});
};
for(const name of ["chat","image","tts","video-start","video-status"]){await import(`../supabase/functions/${name}/index.ts`);handlers[name]=globalThis.latestHandler;}
const id="123e4567-e89b-42d3-a456-426614174000";
function reset(){videoJob=null;mergeClaim=true;videoQueryError=false;settlementError=false;queueCount=0;providerCalls=0;calls=[];replay=null;receiptReason="reserved";reserveAllowed=true;signedFailure=false;storageFailure=false;saveFailure=false;providerOk=true;authenticated=true;verified=true;responseValue={choices:[{message:{content:"Hola"}}]};}
async function request(name,payload){return handlers[name](new Request("https://test.supabase.co/functions/v1/"+name,{method:"POST",headers:{authorization:"Bearer test-session"},body:JSON.stringify({...payload,idempotency_key:id})}));}
const plan={action:"video",message:"Un mensaje de esperanza.",prompt:"A peaceful sunrise over the sea",narration:"Dios va delante de ti. Sigue adelante con fe.",required_features:["video","voice"]};
let passed=0;
async function test(name,run){reset();await run();passed++;console.log("ok",name);}
await test("Brain planner validates and persists its plan",async()=>{
  responseValue={choices:[{message:{content:JSON.stringify(plan)}}]};
  const response=await request("chat",{message:"Crea un video inspirador",purpose:"brain",language:"es"});
  assert.equal(response.status,200);assert.deepEqual((await response.json()).plan,plan);
  const provider=calls.find(c=>c.name==="provider");assert.equal(provider.body.response_format.type,"json_schema"); assert.equal(provider.body.response_format.json_schema.strict,true); assert.equal(provider.body.response_format.json_schema.schema.additionalProperties,false); assert.deepEqual(provider.body.response_format.json_schema.schema.required,["action","message","prompt","narration","required_features"]);
  assert.ok(provider.body.messages[0].content.includes("Spanish"));
  assert.equal(calls.find(c=>c.name==="save_generation_result").args.p_result.plan.action,"video");
  assert.equal(calls.filter(c=>c.name==="finalize_generation").length,0,"Saving result atomically settles the event");
});
await test("Planner refuses unsupported execution and refunds once",async()=>{
  responseValue={choices:[{message:{content:JSON.stringify({...plan,required_features:["video","music"]})}}]};
  const response=await request("chat",{message:"Crea video con música",purpose:"brain"});
  assert.equal(response.status,502);assert.equal(calls.filter(c=>c.name==="finalize_generation"&&c.args.p_success===false).length,1);assert.ok(!calls.some(c=>c.name==="save_generation_result"));
});
await test("Malformed JSON from planner cannot start media",async()=>{
  responseValue={choices:[{message:{content:"not json"}}]};assert.equal((await request("chat",{message:"Crea un video",purpose:"brain"})).status,502);
});
await test("Explicit English output reaches the provider",async()=>{
  await request("chat",{message:"Hola",language:"en"});assert.ok(calls.find(c=>c.name==="provider").body.messages[0].content.includes("Write the result in English"));
});
await test("Repeated completed plan returns saved content without provider",async()=>{
  receiptReason="idempotent_replay";replay={status:"ok",reply:"Guardado",plan};
  const response=await request("chat",{message:"Hola",purpose:"brain"});assert.equal(response.status,200);assert.equal((await response.json()).reply,"Guardado");assert.equal(providerCalls,0);
});
await test("Repeated in-flight request never generates or refunds original",async()=>{
  receiptReason="idempotent_replay";const response=await request("chat",{message:"Hola"});assert.equal(response.status,409);assert.equal(providerCalls,0);assert.ok(!calls.some(c=>c.name==="finalize_generation"));
});
await test("Unverified account cannot reach a paid provider",async()=>{
  verified=false;assert.equal((await request("chat",{message:"Hola"})).status,403);assert.equal(providerCalls,0);assert.ok(!calls.some(c=>c.name.startsWith("reserve")));
});
await test("Unauthenticated request cannot reach a paid provider",async()=>{
  authenticated=false;assert.equal((await request("chat",{message:"Hola"})).status,401);assert.equal(providerCalls,0);
});
await test("Exhausted credits stop before provider",async()=>{
  reserveAllowed=false;assert.equal((await request("chat",{message:"Hola"})).status,402);assert.equal(providerCalls,0);
});
await test("Overlong speech rejected before reservation",async()=>{
  assert.equal((await request("tts",{text:"a".repeat(1201)})).status,400);assert.equal(calls.length,0);
});
await test("Speech preserves literal text and saves actual audio",async()=>{
  responseValue=new Uint8Array([73,68,51]);const response=await request("tts",{text:"Dios va adelante.",voice:"onyx",instructions:"Speak warmly"});
  assert.equal(response.status,200);assert.ok((await response.json()).audio_url.startsWith("https://test.supabase.co/"));
  assert.equal(calls.find(c=>c.name==="provider").body.input,"Dios va adelante.");
  assert.equal(calls.find(c=>c.name==="save_generation_result").args.p_result.media_field,"audio_url");
});
await test("Provider failure refunds and never claims completion",async()=>{
  providerOk=false;assert.equal((await request("tts",{text:"Hola"})).status,500);assert.ok(calls.some(c=>c.name==="finalize_generation"&&c.args.p_success===false));
});
await test("Empty audio is not a successful result",async()=>{
  responseValue=new Uint8Array();assert.equal((await request("tts",{text:"Hola"})).status,500);assert.ok(!calls.some(c=>c.name==="save_generation_result"));
});
await test("Image bytes are stored instead of large database payloads",async()=>{
  responseValue={data:[{b64_json:"iVBORw0KGgo="}]};assert.equal((await request("image",{prompt:"A peaceful sunrise"})).status,200);assert.equal(calls.find(c=>c.name==="save_generation_result").args.p_result.storage_path,"test-reservation/result.png");
});
await test("File storage failure refunds instead of returning broken URL",async()=>{
  responseValue=new Uint8Array([1,2]);storageFailure=true;assert.equal((await request("tts",{text:"Hola"})).status,500);assert.ok(calls.some(c=>c.name==="finalize_generation"&&!c.args.p_success));
});
await test("Replay of media signs a new URL without regenerating",async()=>{
  receiptReason="idempotent_replay";replay={status:"ok",storage_path:"test-reservation/result.mp3",media_field:"audio_url"};const response=await request("tts",{text:"Hola"});assert.equal(response.status,200);assert.ok((await response.json()).audio_url);assert.equal(providerCalls,0);
});
await test("Cached output signing failure never refunds completed result",async()=>{
  receiptReason="idempotent_replay";replay={storage_path:"path",media_field:"audio_url"};signedFailure=true;assert.equal((await request("tts",{text:"Hola"})).status,500);assert.ok(!calls.some(c=>c.name==="finalize_generation"));
});
await test("Result persistence failure cannot mark success",async()=>{
  saveFailure=true;assert.equal((await request("chat",{message:"Hola"})).status,500);assert.ok(calls.some(c=>c.name==="finalize_generation"&&!c.args.p_success));
});
await test("OPTIONS never authenticates, reserves or calls provider",async()=>{
  const response=await handlers.chat(new Request("https://test.supabase.co/chat",{method:"OPTIONS"}));assert.equal(response.status,200);assert.equal(calls.length,0);
});
await test("Non-POST method rejected before dependencies",async()=>{
  assert.equal((await handlers.chat(new Request("https://test.supabase.co/chat"))).status,405);assert.equal(calls.length,0);
});
console.log(`${passed} real Edge-handler integration scenarios passed; every provider request mocked.`);

const makeJob=()=>({id:"video-job",user_id:"test-user",reservation_id:"test-reservation",status:"queued",created_at:new Date().toISOString(),duration_seconds:5,narration_text:"Dios va adelante.",voice:"nova",status_url:"https://queue.fal.run/status",response_url:"https://queue.fal.run/result"});
await test("Video start replay cannot submit a second provider request",async()=>{
  receiptReason="idempotent_replay";
  assert.equal((await request("video-start",{prompt:"A peaceful sunrise",narration:"Dios va adelante.",duration:5})).status,409);
  assert.equal(queueCount,0);assert.ok(!calls.some(c=>c.name==="finalize_generation"));
});
await test("Existing video request returns the same job",async()=>{
  receiptReason="idempotent_replay";videoJob=makeJob();
  const response=await request("video-start",{prompt:"A peaceful sunrise",narration:"Dios va adelante.",duration:5});
  assert.equal(response.status,202);assert.equal((await response.json()).job_id,"video-job");assert.equal(queueCount,0);
});
await test("Unsafe video cannot reserve credits or reach provider",async()=>{
  assert.equal((await request("video-start",{prompt:"pornografia",narration:"Hola",duration:5})).status,400);assert.equal(calls.length,0);assert.equal(queueCount,0);
});
await test("Video status enforces job ownership",async()=>{
  videoJob={...makeJob(),user_id:"other-user"};
  assert.equal((await request("video-status",{job_id:"video-job"})).status,404);assert.equal(queueCount,0);assert.equal(providerCalls,0);
});
await test("Database read failure is temporary, not missing video",async()=>{
  videoJob=makeJob();videoQueryError=true;assert.equal((await request("video-status",{job_id:"video-job"})).status,503);assert.equal(providerCalls,0);
});
await test("Lost settlement cannot falsely claim completion or refund",async()=>{
  videoJob={...makeJob(),status:"completed",video_url:"https://example.invalid/video.mp4"};settlementError=true;
  assert.equal((await request("video-status",{job_id:"video-job"})).status,503);assert.equal(providerCalls,0);
});
await test("Only one concurrent poll can start audio merging",async()=>{
  videoJob=makeJob();responseValue=new Uint8Array(48000);
  const responses=await Promise.all([request("video-status",{job_id:"video-job"}),request("video-status",{job_id:"video-job"})]);
  assert.equal(responses[0].status,202);assert.equal(responses[1].status,202);assert.equal(queueCount,1);assert.equal(providerCalls,1);
  const composition=calls.find(c=>c.name==="fal");assert.equal(composition.url,"https://queue.fal.run/fal-ai/ffmpeg-api/compose");assert.equal(composition.body.tracks[0].keyframes[0].duration,5000);assert.equal(composition.body.tracks[1].keyframes[0].timestamp,0);assert.equal(composition.body.tracks[1].keyframes[0].duration,1000);assert.equal(calls.find(c=>c.name==="provider").body.response_format,"pcm");
});
await test("Video output and credits settle through one database transaction",async()=>{
  videoJob={...makeJob(),merge_request_id:"merge",merge_status_url:"https://queue.fal.run/status",merge_response_url:"https://queue.fal.run/result"};
  const response=await request("video-status",{job_id:"video-job"});assert.equal(response.status,200);assert.equal((await response.json()).status,"completed");
  assert.equal(calls.filter(c=>c.name==="settle_video_job").length,1);assert.ok(!calls.some(c=>c.name==="finalize_generation"));
});
await test("Expired video returns only a confirmed refund",async()=>{
  videoJob={...makeJob(),created_at:"2020-01-01T00:00:00Z"};const response=await request("video-status",{job_id:"video-job"});assert.equal((await response.json()).refunded,true);assert.equal(queueCount,0);
});
console.log(`${passed} Edge integration scenarios passed, including video dependencies.`);

await test("Oversized unknown input cannot exhaust planner memory or budget",async()=>{
  const response=await request("chat",{message:"Hola",purpose:"brain",preferences:{extra:"x".repeat(20000)}});
  assert.equal(response.status,413);assert.equal(calls.length,0);assert.equal(providerCalls,0);
});
await test("Malformed or non-object JSON cannot reserve credits",async()=>{
  for(const body of ["not json","[]","null"]){
    const response=await handlers.chat(new Request("https://test.supabase.co/chat",{method:"POST",body}));
    assert.equal(response.status,400);
  }
  assert.equal(calls.length,0);
});
await test("English explicit content is blocked on the server",async()=>{
  const response=await request("image",{prompt:"Generate a nude explicit image"});
  assert.equal(response.status,400);assert.equal(calls.length,0);
});
console.log(`${passed} complete Edge scenarios passed; no external provider was contacted.`);

await test("Invalid PCM cannot submit composition or claim success",async()=>{
  videoJob=makeJob();responseValue=new Uint8Array(3);
  const response=await request("video-status",{job_id:"video-job"});assert.equal((await response.json()).status,"failed");assert.equal(queueCount,0);assert.equal(calls.find(c=>c.name==="settle_video_job").args.p_success,false);
});
