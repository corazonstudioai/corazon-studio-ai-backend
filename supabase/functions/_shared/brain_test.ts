import { assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { brainInstructions, parseBrainPlan, parseBrainReply } from "./brain.ts";

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
