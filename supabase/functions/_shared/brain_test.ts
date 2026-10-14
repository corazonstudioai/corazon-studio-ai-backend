import { assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { brainInstructions, parseBrainPlan, parseBrainReply, validateBrainIntent, wantsSilentVideo } from "./brain.ts";

const text={action:"text",message:"Dios va adelante.",prompt:"",narration:"",required_features:["text"]};
const video={action:"video",message:"Un video",prompt:"Un amanecer tranquilo",narration:"Dios va adelante. Sigue con fe.",required_features:["video","voice"]};
Deno.test("Brain instructions bind language and capabilities",()=>{
  assertEquals(brainInstructions("en").includes("English"),true);
  assertEquals(brainInstructions("es").includes("Spanish"),true);
  assertEquals(brainInstructions("es").includes("NOT available yet"),true);
});
Deno.test("Brain accepts finished text and bounded video narration",()=>{
  assertEquals(parseBrainPlan(text),text);
  assertEquals(parseBrainReply(JSON.stringify(video)),video);
});
Deno.test("Brain can explain unavailable features without pretending to execute",()=>{
  assertEquals(parseBrainPlan({...text,action:"clarify",required_features:["music"]}).action,"clarify");
  assertThrows(()=>parseBrainPlan({...video,required_features:["music"]}));
});
Deno.test("Brain rejects malformed JSON, blank and overlong fields",()=>{
  for(const value of [null,{},[],{...text,message:""},{...video,prompt:""},{...video,narration:""},{...text,message:"x".repeat(4001)},{...video,narration:"word ".repeat(13)},{...text,required_features:"text"}])assertThrows(()=>parseBrainPlan(value));
  assertThrows(()=>parseBrainReply("not JSON"));
});

Deno.test("silent intent is validated independently from the model",()=>{
  const silent={...video,narration:"",required_features:["video"],audio_mode:"silent"};
  assertEquals(parseBrainPlan(silent).audio_mode,"silent");
  for(const request of ["Crea un video sin voz","sin texto ni narración","video sin audio","without narration","no voiceover","silent video","sin música, texto ni narración","without any narration","without on-screen text or voice-over","sin sonidos"]){assertEquals(wantsSilentVideo(request),true);validateBrainIntent(parseBrainPlan(silent),request,"video");assertThrows(()=>validateBrainIntent(video,request));}
  assertEquals(wantsSilentVideo("Crea video con voz cálida"),false);
  assertThrows(()=>parseBrainPlan({...silent,narration:"No debes hablar"}));
  assertThrows(()=>parseBrainPlan({...silent,required_features:["video","voice"]}));
  assertThrows(()=>parseBrainPlan({...text,audio_mode:"silent"}));
  assertThrows(()=>validateBrainIntent(text,"Una montaña","video"));
});
