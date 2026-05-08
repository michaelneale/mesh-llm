import{g as te}from"./chunk-55IACEB6-DoYbnq4t.js";import{s as ee}from"./chunk-EDXVE4YY-BdB1vSfN.js";import{_ as f,l as b,c as F,r as se,u as ie,a as re,b as ae,g as ne,s as le,p as oe,q as ce,T as he,k as H,z as ue}from"./mermaid.core-DcBdMCAd.js";var vt=function(){var t=f(function(V,l,h,n){for(h=h||{},n=V.length;n--;h[V[n]]=l);return h},"o"),e=[1,2],s=[1,3],a=[1,4],i=[2,4],o=[1,9],u=[1,11],p=[1,16],S=[1,17],T=[1,18],E=[1,19],m=[1,33],L=[1,20],v=[1,21],O=[1,22],w=[1,23],C=[1,24],d=[1,26],D=[1,27],$=[1,28],B=[1,29],I=[1,30],G=[1,31],P=[1,32],at=[1,35],nt=[1,36],lt=[1,37],ot=[1,38],K=[1,34],y=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],ct=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],xt=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],gt={trace:f(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"CLICK",39:"STRING",40:"HREF",41:"classDef",42:"CLASSDEF_ID",43:"CLASSDEF_STYLEOPTS",44:"DEFAULT",45:"style",46:"STYLE_IDS",47:"STYLEDEF_STYLEOPTS",48:"class",49:"CLASSENTITY_IDS",50:"STYLECLASS",51:"direction_tb",52:"direction_bt",53:"direction_rl",54:"direction_lr",56:";",57:"EDGE_STATE",58:"STYLE_SEPARATOR",59:"left_of",60:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:f(function(l,h,n,g,_,r,Z){var c=r.length-1;switch(_){case 3:return g.setRootDoc(r[c]),r[c];case 4:this.$=[];break;case 5:r[c]!="nl"&&(r[c-1].push(r[c]),this.$=r[c-1]);break;case 6:case 7:this.$=r[c];break;case 8:this.$="nl";break;case 12:this.$=r[c];break;case 13:const tt=r[c-1];tt.description=g.trimColon(r[c]),this.$=tt;break;case 14:this.$={stmt:"relation",state1:r[c-2],state2:r[c]};break;case 15:const Tt=g.trimColon(r[c]);this.$={stmt:"relation",state1:r[c-3],state2:r[c-1],description:Tt};break;case 19:this.$={stmt:"state",id:r[c-3],type:"default",description:"",doc:r[c-1]};break;case 20:var U=r[c],X=r[c-2].trim();if(r[c].match(":")){var ut=r[c].split(":");U=ut[0],X=[X,ut[1]]}this.$={stmt:"state",id:U,type:"default",description:X};break;case 21:this.$={stmt:"state",id:r[c-3],type:"default",description:r[c-5],doc:r[c-1]};break;case 22:this.$={stmt:"state",id:r[c],type:"fork"};break;case 23:this.$={stmt:"state",id:r[c],type:"join"};break;case 24:this.$={stmt:"state",id:r[c],type:"choice"};break;case 25:this.$={stmt:"state",id:g.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:r[c-1].trim(),note:{position:r[c-2].trim(),text:r[c].trim()}};break;case 29:this.$=r[c].trim(),g.setAccTitle(this.$);break;case 30:case 31:this.$=r[c].trim(),g.setAccDescription(this.$);break;case 32:this.$={stmt:"click",id:r[c-3],url:r[c-2],tooltip:r[c-1]};break;case 33:this.$={stmt:"click",id:r[c-3],url:r[c-1],tooltip:""};break;case 34:case 35:this.$={stmt:"classDef",id:r[c-1].trim(),classes:r[c].trim()};break;case 36:this.$={stmt:"style",id:r[c-1].trim(),styleClass:r[c].trim()};break;case 37:this.$={stmt:"applyClass",id:r[c-1].trim(),styleClass:r[c].trim()};break;case 38:g.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 39:g.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 40:g.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 41:g.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 44:case 45:this.$={stmt:"state",id:r[c].trim(),type:"default",description:""};break;case 46:this.$={stmt:"state",id:r[c-2].trim(),classes:[r[c].trim()],type:"default",description:""};break;case 47:this.$={stmt:"state",id:r[c-2].trim(),classes:[r[c].trim()],type:"default",description:""};break}},"anonymous"),table:[{3:1,4:e,5:s,6:a},{1:[3]},{3:5,4:e,5:s,6:a},{3:6,4:e,5:s,6:a},t([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],i,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:o,5:u,8:8,9:10,10:12,11:13,12:14,13:15,16:p,17:S,19:T,22:E,24:m,25:L,26:v,27:O,28:w,29:C,32:25,33:d,35:D,37:$,38:B,41:I,45:G,48:P,51:at,52:nt,53:lt,54:ot,57:K},t(y,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:p,17:S,19:T,22:E,24:m,25:L,26:v,27:O,28:w,29:C,32:25,33:d,35:D,37:$,38:B,41:I,45:G,48:P,51:at,52:nt,53:lt,54:ot,57:K},t(y,[2,7]),t(y,[2,8]),t(y,[2,9]),t(y,[2,10]),t(y,[2,11]),t(y,[2,12],{14:[1,40],15:[1,41]}),t(y,[2,16]),{18:[1,42]},t(y,[2,18],{20:[1,43]}),{23:[1,44]},t(y,[2,22]),t(y,[2,23]),t(y,[2,24]),t(y,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},t(y,[2,28]),{34:[1,49]},{36:[1,50]},t(y,[2,31]),{13:51,24:m,57:K},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},t(ct,[2,44],{58:[1,56]}),t(ct,[2,45],{58:[1,57]}),t(y,[2,38]),t(y,[2,39]),t(y,[2,40]),t(y,[2,41]),t(y,[2,6]),t(y,[2,13]),{13:58,24:m,57:K},t(y,[2,17]),t(xt,i,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},t(y,[2,29]),t(y,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},t(y,[2,14],{14:[1,71]}),{4:o,5:u,8:8,9:10,10:12,11:13,12:14,13:15,16:p,17:S,19:T,21:[1,72],22:E,24:m,25:L,26:v,27:O,28:w,29:C,32:25,33:d,35:D,37:$,38:B,41:I,45:G,48:P,51:at,52:nt,53:lt,54:ot,57:K},t(y,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},t(y,[2,34]),t(y,[2,35]),t(y,[2,36]),t(y,[2,37]),t(ct,[2,46]),t(ct,[2,47]),t(y,[2,15]),t(y,[2,19]),t(xt,i,{7:78}),t(y,[2,26]),t(y,[2,27]),{5:[1,79]},{5:[1,80]},{4:o,5:u,8:8,9:10,10:12,11:13,12:14,13:15,16:p,17:S,19:T,21:[1,81],22:E,24:m,25:L,26:v,27:O,28:w,29:C,32:25,33:d,35:D,37:$,38:B,41:I,45:G,48:P,51:at,52:nt,53:lt,54:ot,57:K},t(y,[2,32]),t(y,[2,33]),t(y,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:f(function(l,h){if(h.recoverable)this.trace(l);else{var n=new Error(l);throw n.hash=h,n}},"parseError"),parse:f(function(l){var h=this,n=[0],g=[],_=[null],r=[],Z=this.table,c="",U=0,X=0,ut=2,tt=1,Tt=r.slice.call(arguments,1),k=Object.create(this.lexer),W={yy:{}};for(var Et in this.yy)Object.prototype.hasOwnProperty.call(this.yy,Et)&&(W.yy[Et]=this.yy[Et]);k.setInput(l,W.yy),W.yy.lexer=k,W.yy.parser=this,typeof k.yylloc>"u"&&(k.yylloc={});var _t=k.yylloc;r.push(_t);var Qt=k.options&&k.options.ranges;typeof W.yy.parseError=="function"?this.parseError=W.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function Zt(R){n.length=n.length-2*R,_.length=_.length-R,r.length=r.length-R}f(Zt,"popStack");function Lt(){var R;return R=g.pop()||k.lex()||tt,typeof R!="number"&&(R instanceof Array&&(g=R,R=g.pop()),R=h.symbols_[R]||R),R}f(Lt,"lex");for(var x,j,N,mt,J={},dt,Y,Ot,ft;;){if(j=n[n.length-1],this.defaultActions[j]?N=this.defaultActions[j]:((x===null||typeof x>"u")&&(x=Lt()),N=Z[j]&&Z[j][x]),typeof N>"u"||!N.length||!N[0]){var bt="";ft=[];for(dt in Z[j])this.terminals_[dt]&&dt>ut&&ft.push("'"+this.terminals_[dt]+"'");k.showPosition?bt="Parse error on line "+(U+1)+`:
`+k.showPosition()+`
Expecting `+ft.join(", ")+", got '"+(this.terminals_[x]||x)+"'":bt="Parse error on line "+(U+1)+": Unexpected "+(x==tt?"end of input":"'"+(this.terminals_[x]||x)+"'"),this.parseError(bt,{text:k.match,token:this.terminals_[x]||x,line:k.yylineno,loc:_t,expected:ft})}if(N[0]instanceof Array&&N.length>1)throw new Error("Parse Error: multiple actions possible at state: "+j+", token: "+x);switch(N[0]){case 1:n.push(x),_.push(k.yytext),r.push(k.yylloc),n.push(N[1]),x=null,X=k.yyleng,c=k.yytext,U=k.yylineno,_t=k.yylloc;break;case 2:if(Y=this.productions_[N[1]][1],J.$=_[_.length-Y],J._$={first_line:r[r.length-(Y||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(Y||1)].first_column,last_column:r[r.length-1].last_column},Qt&&(J._$.range=[r[r.length-(Y||1)].range[0],r[r.length-1].range[1]]),mt=this.performAction.apply(J,[c,X,U,W.yy,N[1],_,r].concat(Tt)),typeof mt<"u")return mt;Y&&(n=n.slice(0,-1*Y*2),_=_.slice(0,-1*Y),r=r.slice(0,-1*Y)),n.push(this.productions_[N[1]][0]),_.push(J.$),r.push(J._$),Ot=Z[n[n.length-2]][n[n.length-1]],n.push(Ot);break;case 3:return!0}}return!0},"parse")},qt=function(){var V={EOF:1,parseError:f(function(h,n){if(this.yy.parser)this.yy.parser.parseError(h,n);else throw new Error(h)},"parseError"),setInput:f(function(l,h){return this.yy=h||this.yy||{},this._input=l,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:f(function(){var l=this._input[0];this.yytext+=l,this.yyleng++,this.offset++,this.match+=l,this.matched+=l;var h=l.match(/(?:\r\n?|\n).*/g);return h?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),l},"input"),unput:f(function(l){var h=l.length,n=l.split(/(?:\r\n?|\n)/g);this._input=l+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-h),this.offset-=h;var g=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),n.length-1&&(this.yylineno-=n.length-1);var _=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:n?(n.length===g.length?this.yylloc.first_column:0)+g[g.length-n.length].length-n[0].length:this.yylloc.first_column-h},this.options.ranges&&(this.yylloc.range=[_[0],_[0]+this.yyleng-h]),this.yyleng=this.yytext.length,this},"unput"),more:f(function(){return this._more=!0,this},"more"),reject:f(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:f(function(l){this.unput(this.match.slice(l))},"less"),pastInput:f(function(){var l=this.matched.substr(0,this.matched.length-this.match.length);return(l.length>20?"...":"")+l.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:f(function(){var l=this.match;return l.length<20&&(l+=this._input.substr(0,20-l.length)),(l.substr(0,20)+(l.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:f(function(){var l=this.pastInput(),h=new Array(l.length+1).join("-");return l+this.upcomingInput()+`
`+h+"^"},"showPosition"),test_match:f(function(l,h){var n,g,_;if(this.options.backtrack_lexer&&(_={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(_.yylloc.range=this.yylloc.range.slice(0))),g=l[0].match(/(?:\r\n?|\n).*/g),g&&(this.yylineno+=g.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:g?g[g.length-1].length-g[g.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+l[0].length},this.yytext+=l[0],this.match+=l[0],this.matches=l,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(l[0].length),this.matched+=l[0],n=this.performAction.call(this,this.yy,this,h,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),n)return n;if(this._backtrack){for(var r in _)this[r]=_[r];return!1}return!1},"test_match"),next:f(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var l,h,n,g;this._more||(this.yytext="",this.match="");for(var _=this._currentRules(),r=0;r<_.length;r++)if(n=this._input.match(this.rules[_[r]]),n&&(!h||n[0].length>h[0].length)){if(h=n,g=r,this.options.backtrack_lexer){if(l=this.test_match(n,_[r]),l!==!1)return l;if(this._backtrack){h=!1;continue}else return!1}else if(!this.options.flex)break}return h?(l=this.test_match(h,_[g]),l!==!1?l:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:f(function(){var h=this.next();return h||this.lex()},"lex"),begin:f(function(h){this.conditionStack.push(h)},"begin"),popState:f(function(){var h=this.conditionStack.length-1;return h>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:f(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:f(function(h){return h=this.conditionStack.length-1-Math.abs(h||0),h>=0?this.conditionStack[h]:"INITIAL"},"topState"),pushState:f(function(h){this.begin(h)},"pushState"),stateStackSize:f(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:f(function(h,n,g,_){switch(g){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:return 51;case 5:return 52;case 6:return 53;case 7:return 54;case 8:break;case 9:break;case 10:return 5;case 11:break;case 12:break;case 13:break;case 14:break;case 15:return this.pushState("SCALE"),17;case 16:return 18;case 17:this.popState();break;case 18:return this.begin("acc_title"),33;case 19:return this.popState(),"acc_title_value";case 20:return this.begin("acc_descr"),35;case 21:return this.popState(),"acc_descr_value";case 22:this.begin("acc_descr_multiline");break;case 23:this.popState();break;case 24:return"acc_descr_multiline_value";case 25:return this.pushState("CLASSDEF"),41;case 26:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";case 27:return this.popState(),this.pushState("CLASSDEFID"),42;case 28:return this.popState(),43;case 29:return this.pushState("CLASS"),48;case 30:return this.popState(),this.pushState("CLASS_STYLE"),49;case 31:return this.popState(),50;case 32:return this.pushState("STYLE"),45;case 33:return this.popState(),this.pushState("STYLEDEF_STYLES"),46;case 34:return this.popState(),47;case 35:return this.pushState("SCALE"),17;case 36:return 18;case 37:this.popState();break;case 38:this.pushState("STATE");break;case 39:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),25;case 40:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),26;case 41:return this.popState(),n.yytext=n.yytext.slice(0,-10).trim(),27;case 42:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),25;case 43:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),26;case 44:return this.popState(),n.yytext=n.yytext.slice(0,-10).trim(),27;case 45:return 51;case 46:return 52;case 47:return 53;case 48:return 54;case 49:this.pushState("STATE_STRING");break;case 50:return this.pushState("STATE_ID"),"AS";case 51:return this.popState(),"ID";case 52:this.popState();break;case 53:return"STATE_DESCR";case 54:return 19;case 55:this.popState();break;case 56:return this.popState(),this.pushState("struct"),20;case 57:break;case 58:return this.popState(),21;case 59:break;case 60:return this.begin("NOTE"),29;case 61:return this.popState(),this.pushState("NOTE_ID"),59;case 62:return this.popState(),this.pushState("NOTE_ID"),60;case 63:this.popState(),this.pushState("FLOATING_NOTE");break;case 64:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";case 65:break;case 66:return"NOTE_TEXT";case 67:return this.popState(),"ID";case 68:return this.popState(),this.pushState("NOTE_TEXT"),24;case 69:return this.popState(),n.yytext=n.yytext.substr(2).trim(),31;case 70:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),31;case 71:return 6;case 72:return 6;case 73:return 16;case 74:return 57;case 75:return 24;case 76:return n.yytext=n.yytext.trim(),14;case 77:return 15;case 78:return 28;case 79:return 58;case 80:return 5;case 81:return"INVALID"}},"anonymous"),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:(?:[^:\n;]|:[^:\n;])+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[12,13],inclusive:!1},struct:{rules:[12,13,25,29,32,38,45,46,47,48,57,58,59,60,74,75,76,77,78,79],inclusive:!1},FLOATING_NOTE_ID:{rules:[67],inclusive:!1},FLOATING_NOTE:{rules:[64,65,66],inclusive:!1},NOTE_TEXT:{rules:[69,70],inclusive:!1},NOTE_ID:{rules:[68],inclusive:!1},NOTE:{rules:[61,62,63],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[34],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[33],inclusive:!1},CLASS_STYLE:{rules:[31],inclusive:!1},CLASS:{rules:[30],inclusive:!1},CLASSDEFID:{rules:[28],inclusive:!1},CLASSDEF:{rules:[26,27],inclusive:!1},acc_descr_multiline:{rules:[23,24],inclusive:!1},acc_descr:{rules:[21],inclusive:!1},acc_title:{rules:[19],inclusive:!1},SCALE:{rules:[16,17,36,37],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[51],inclusive:!1},STATE_STRING:{rules:[52,53],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[12,13,39,40,41,42,43,44,49,50,54,55,56],inclusive:!1},ID:{rules:[12,13],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,18,20,22,25,29,32,35,38,56,60,71,72,73,74,75,76,77,79,80,81],inclusive:!0}}};return V}();gt.lexer=qt;function ht(){this.yy={}}return f(ht,"Parser"),ht.prototype=gt,gt.Parser=ht,new ht}();vt.parser=vt;var Ye=vt,de="TB",Bt="TB",Rt="dir",Q="state",q="root",Ct="relation",fe="classDef",pe="style",Se="applyClass",it="default",Gt="divider",Yt="fill:none",Vt="fill: #333",Mt="c",Ut="markdown",Wt="normal",kt="rect",Dt="rectWithTitle",ye="stateStart",ge="stateEnd",It="divider",Nt="roundedWithTitle",Te="note",Ee="noteGroup",rt="statediagram",_e="state",me=`${rt}-${_e}`,jt="transition",be="note",ke="note-edge",De=`${jt} ${ke}`,ve=`${rt}-${be}`,Ce="cluster",Ae=`${rt}-${Ce}`,xe="cluster-alt",Le=`${rt}-${xe}`,Ht="parent",zt="note",Oe="state",At="----",Re=`${At}${zt}`,wt=`${At}${Ht}`,Kt=f((t,e=Bt)=>{if(!t.doc)return e;let s=e;for(const a of t.doc)a.stmt==="dir"&&(s=a.value);return s},"getDir"),Ie=f(function(t,e){return e.db.getClasses()},"getClasses"),Ne=f(async function(t,e,s,a){b.info("REF0:"),b.info("Drawing state diagram (v2)",e);const{securityLevel:i,state:o,layout:u}=F();a.db.extract(a.db.getRootDocV2());const p=a.db.getData(),S=te(e,i);p.type=a.type,p.layoutAlgorithm=u,p.nodeSpacing=(o==null?void 0:o.nodeSpacing)||50,p.rankSpacing=(o==null?void 0:o.rankSpacing)||50,F().look==="neo"?p.markers=["barbNeo"]:p.markers=["barb"],p.diagramId=e,await se(p,S);const E=8;try{(typeof a.db.getLinks=="function"?a.db.getLinks():new Map).forEach((L,v)=>{var B;const O=typeof v=="string"?v:typeof(v==null?void 0:v.id)=="string"?v.id:"";if(!O){b.warn("⚠️ Invalid or missing stateId from key:",JSON.stringify(v));return}const w=(B=S.node())==null?void 0:B.querySelectorAll("g");let C;if(w==null||w.forEach(I=>{var P;((P=I.textContent)==null?void 0:P.trim())===O&&(C=I)}),!C){b.warn("⚠️ Could not find node matching text:",O);return}const d=C.parentNode;if(!d){b.warn("⚠️ Node has no parent, cannot wrap:",O);return}const D=document.createElementNS("http://www.w3.org/2000/svg","a"),$=L.url.replace(/^"+|"+$/g,"");if(D.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",$),D.setAttribute("target","_blank"),L.tooltip){const I=L.tooltip.replace(/^"+|"+$/g,"");D.setAttribute("title",I)}d.replaceChild(D,C),D.appendChild(C),b.info("🔗 Wrapped node in <a> tag for:",O,L.url)})}catch(m){b.error("❌ Error injecting clickable links:",m)}ie.insertTitle(S,"statediagramTitleText",(o==null?void 0:o.titleTopMargin)??25,a.db.getDiagramTitle()),ee(S,E,rt,(o==null?void 0:o.useMaxWidth)??!0)},"draw"),Ve={getClasses:Ie,draw:Ne,getDir:Kt},St=new Map,M=0;function yt(t="",e=0,s="",a=At){const i=s!==null&&s.length>0?`${a}${s}`:"";return`${Oe}-${t}${i}-${e}`}f(yt,"stateDomId");var we=f((t,e,s,a,i,o,u,p)=>{b.trace("items",e),e.forEach(S=>{switch(S.stmt){case Q:st(t,S,s,a,i,o,u,p);break;case it:st(t,S,s,a,i,o,u,p);break;case Ct:{st(t,S.state1,s,a,i,o,u,p),st(t,S.state2,s,a,i,o,u,p);const T=u==="neo",E={id:"edge"+M,start:S.state1.id,end:S.state2.id,arrowhead:"normal",arrowTypeEnd:T?"arrow_barb_neo":"arrow_barb",style:Yt,labelStyle:"",label:H.sanitizeText(S.description??"",F()),arrowheadStyle:Vt,labelpos:Mt,labelType:Ut,thickness:Wt,classes:jt,look:u};i.push(E),M++}break}})},"setupDoc"),$t=f((t,e=Bt)=>{let s=e;if(t.doc)for(const a of t.doc)a.stmt==="dir"&&(s=a.value);return s},"getDir");function et(t,e,s){if(!e.id||e.id==="</join></fork>"||e.id==="</choice>")return;e.cssClasses&&(Array.isArray(e.cssCompiledStyles)||(e.cssCompiledStyles=[]),e.cssClasses.split(" ").forEach(i=>{const o=s.get(i);o&&(e.cssCompiledStyles=[...e.cssCompiledStyles??[],...o.styles])}));const a=t.find(i=>i.id===e.id);a?Object.assign(a,e):t.push(e)}f(et,"insertOrUpdateNode");function Xt(t){var e;return((e=t==null?void 0:t.classes)==null?void 0:e.join(" "))??""}f(Xt,"getClassesFromDbInfo");function Jt(t){return(t==null?void 0:t.styles)??[]}f(Jt,"getStylesFromDbInfo");var st=f((t,e,s,a,i,o,u,p)=>{var v,O,w;const S=e.id,T=s.get(S),E=Xt(T),m=Jt(T),L=F();if(b.info("dataFetcher parsedItem",e,T,m),S!=="root"){let C=kt;e.start===!0?C=ye:e.start===!1&&(C=ge),e.type!==it&&(C=e.type),St.get(S)||St.set(S,{id:S,shape:C,description:H.sanitizeText(S,L),cssClasses:`${E} ${me}`,cssStyles:m});const d=St.get(S);e.description&&(Array.isArray(d.description)?(d.shape=Dt,d.description.push(e.description)):(v=d.description)!=null&&v.length&&d.description.length>0?(d.shape=Dt,d.description===S?d.description=[e.description]:d.description=[d.description,e.description]):(d.shape=kt,d.description=e.description),d.description=H.sanitizeTextOrArray(d.description,L)),((O=d.description)==null?void 0:O.length)===1&&d.shape===Dt&&(d.type==="group"?d.shape=Nt:d.shape=kt),!d.type&&e.doc&&(b.info("Setting cluster for XCX",S,$t(e)),d.type="group",d.isGroup=!0,d.dir=$t(e),d.shape=e.type===Gt?It:Nt,d.cssClasses=`${d.cssClasses} ${Ae} ${o?Le:""}`);const D={labelStyle:"",shape:d.shape,label:d.description,cssClasses:d.cssClasses,cssCompiledStyles:[],cssStyles:d.cssStyles,id:S,dir:d.dir,domId:yt(S,M),type:d.type,isGroup:d.type==="group",padding:8,rx:10,ry:10,look:u,labelType:"markdown"};if(D.shape===It&&(D.label=""),t&&t.id!=="root"&&(b.trace("Setting node ",S," to be child of its parent ",t.id),D.parentId=t.id),D.centerLabel=!0,e.note){const $={labelStyle:"",shape:Te,label:e.note.text,labelType:"markdown",cssClasses:ve,cssStyles:[],cssCompiledStyles:[],id:S+Re+"-"+M,domId:yt(S,M,zt),type:d.type,isGroup:d.type==="group",padding:(w=L.flowchart)==null?void 0:w.padding,look:u,position:e.note.position},B=S+wt,I={labelStyle:"",shape:Ee,label:e.note.text,cssClasses:d.cssClasses,cssStyles:[],id:S+wt,domId:yt(S,M,Ht),type:"group",isGroup:!0,padding:16,look:u,position:e.note.position};M++,I.id=B,$.parentId=B,et(a,I,p),et(a,$,p),et(a,D,p);let G=S,P=$.id;e.note.position==="left of"&&(G=$.id,P=S),i.push({id:G+"-"+P,start:G,end:P,arrowhead:"none",arrowTypeEnd:"",style:Yt,labelStyle:"",classes:De,arrowheadStyle:Vt,labelpos:Mt,labelType:Ut,thickness:Wt,look:u})}else et(a,D,p)}e.doc&&(b.trace("Adding nodes children "),we(e,e.doc,s,a,i,!o,u,p))},"dataFetcher"),$e=f(()=>{St.clear(),M=0},"reset"),A={START_NODE:"[*]",START_TYPE:"start",END_NODE:"[*]",END_TYPE:"end",COLOR_KEYWORD:"color",FILL_KEYWORD:"fill",BG_FILL:"bgFill",STYLECLASS_SEP:","},Pt=f(()=>new Map,"newClassesList"),Ft=f(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),pt=f(t=>JSON.parse(JSON.stringify(t)),"clone"),z,Me=(z=class{constructor(e){this.version=e,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=Pt(),this.documents={root:Ft()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.getAccTitle=re,this.setAccTitle=ae,this.getAccDescription=ne,this.setAccDescription=le,this.setDiagramTitle=oe,this.getDiagramTitle=ce,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this)}extract(e){this.clear(!0);for(const i of Array.isArray(e)?e:e.doc)switch(i.stmt){case Q:this.addState(i.id.trim(),i.type,i.doc,i.description,i.note);break;case Ct:this.addRelation(i.state1,i.state2,i.description);break;case fe:this.addStyleClass(i.id.trim(),i.classes);break;case pe:this.handleStyleDef(i);break;case Se:this.setCssClass(i.id.trim(),i.styleClass);break;case"click":this.addLink(i.id,i.url,i.tooltip);break}const s=this.getStates(),a=F();$e(),st(void 0,this.getRootDocV2(),s,this.nodes,this.edges,!0,a.look,this.classes);for(const i of this.nodes)if(Array.isArray(i.label)){if(i.description=i.label.slice(1),i.isGroup&&i.description.length>0)throw new Error(`Group nodes can only have label. Remove the additional description for node [${i.id}]`);i.label=i.label[0]}}handleStyleDef(e){const s=e.id.trim().split(","),a=e.styleClass.split(",");for(const i of s){let o=this.getState(i);if(!o){const u=i.trim();this.addState(u),o=this.getState(u)}o&&(o.styles=a.map(u=>{var p;return(p=u.replace(/;/g,""))==null?void 0:p.trim()}))}}setRootDoc(e){b.info("Setting root doc",e),this.rootDoc=e,this.version===1?this.extract(e):this.extract(this.getRootDocV2())}docTranslator(e,s,a){if(s.stmt===Ct){this.docTranslator(e,s.state1,!0),this.docTranslator(e,s.state2,!1);return}if(s.stmt===Q&&(s.id===A.START_NODE?(s.id=e.id+(a?"_start":"_end"),s.start=a):s.id=s.id.trim()),s.stmt!==q&&s.stmt!==Q||!s.doc)return;const i=[];let o=[];for(const u of s.doc)if(u.type===Gt){const p=pt(u);p.doc=pt(o),i.push(p),o=[]}else o.push(u);if(i.length>0&&o.length>0){const u={stmt:Q,id:he(),type:"divider",doc:pt(o)};i.push(pt(u)),s.doc=i}s.doc.forEach(u=>this.docTranslator(s,u,!0))}getRootDocV2(){return this.docTranslator({id:q,stmt:q},{id:q,stmt:q,doc:this.rootDoc},!0),{id:q,doc:this.rootDoc}}addState(e,s=it,a=void 0,i=void 0,o=void 0,u=void 0,p=void 0,S=void 0){const T=e==null?void 0:e.trim();if(!this.currentDocument.states.has(T))b.info("Adding state ",T,i),this.currentDocument.states.set(T,{stmt:Q,id:T,descriptions:[],type:s,doc:a,note:o,classes:[],styles:[],textStyles:[]});else{const E=this.currentDocument.states.get(T);if(!E)throw new Error(`State not found: ${T}`);E.doc||(E.doc=a),E.type||(E.type=s)}if(i&&(b.info("Setting state description",T,i),(Array.isArray(i)?i:[i]).forEach(m=>this.addDescription(T,m.trim()))),o){const E=this.currentDocument.states.get(T);if(!E)throw new Error(`State not found: ${T}`);E.note=o,E.note.text=H.sanitizeText(E.note.text,F())}u&&(b.info("Setting state classes",T,u),(Array.isArray(u)?u:[u]).forEach(m=>this.setCssClass(T,m.trim()))),p&&(b.info("Setting state styles",T,p),(Array.isArray(p)?p:[p]).forEach(m=>this.setStyle(T,m.trim()))),S&&(b.info("Setting state styles",T,p),(Array.isArray(S)?S:[S]).forEach(m=>this.setTextStyle(T,m.trim())))}clear(e){this.nodes=[],this.edges=[],this.documents={root:Ft()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=Pt(),e||(this.links=new Map,ue())}getState(e){return this.currentDocument.states.get(e)}getStates(){return this.currentDocument.states}logDocuments(){b.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}addLink(e,s,a){this.links.set(e,{url:s,tooltip:a}),b.warn("Adding link",e,s,a)}getLinks(){return this.links}startIdIfNeeded(e=""){return e===A.START_NODE?(this.startEndCount++,`${A.START_TYPE}${this.startEndCount}`):e}startTypeIfNeeded(e="",s=it){return e===A.START_NODE?A.START_TYPE:s}endIdIfNeeded(e=""){return e===A.END_NODE?(this.startEndCount++,`${A.END_TYPE}${this.startEndCount}`):e}endTypeIfNeeded(e="",s=it){return e===A.END_NODE?A.END_TYPE:s}addRelationObjs(e,s,a=""){const i=this.startIdIfNeeded(e.id.trim()),o=this.startTypeIfNeeded(e.id.trim(),e.type),u=this.startIdIfNeeded(s.id.trim()),p=this.startTypeIfNeeded(s.id.trim(),s.type);this.addState(i,o,e.doc,e.description,e.note,e.classes,e.styles,e.textStyles),this.addState(u,p,s.doc,s.description,s.note,s.classes,s.styles,s.textStyles),this.currentDocument.relations.push({id1:i,id2:u,relationTitle:H.sanitizeText(a,F())})}addRelation(e,s,a){if(typeof e=="object"&&typeof s=="object")this.addRelationObjs(e,s,a);else if(typeof e=="string"&&typeof s=="string"){const i=this.startIdIfNeeded(e.trim()),o=this.startTypeIfNeeded(e),u=this.endIdIfNeeded(s.trim()),p=this.endTypeIfNeeded(s);this.addState(i,o),this.addState(u,p),this.currentDocument.relations.push({id1:i,id2:u,relationTitle:a?H.sanitizeText(a,F()):void 0})}}addDescription(e,s){var o;const a=this.currentDocument.states.get(e),i=s.startsWith(":")?s.replace(":","").trim():s;(o=a==null?void 0:a.descriptions)==null||o.push(H.sanitizeText(i,F()))}cleanupLabel(e){return e.startsWith(":")?e.slice(2).trim():e.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(e,s=""){this.classes.has(e)||this.classes.set(e,{id:e,styles:[],textStyles:[]});const a=this.classes.get(e);s&&a&&s.split(A.STYLECLASS_SEP).forEach(i=>{const o=i.replace(/([^;]*);/,"$1").trim();if(RegExp(A.COLOR_KEYWORD).exec(i)){const p=o.replace(A.FILL_KEYWORD,A.BG_FILL).replace(A.COLOR_KEYWORD,A.FILL_KEYWORD);a.textStyles.push(p)}a.styles.push(o)})}getClasses(){return this.classes}setCssClass(e,s){e.split(",").forEach(a=>{var o;let i=this.getState(a);if(!i){const u=a.trim();this.addState(u),i=this.getState(u)}(o=i==null?void 0:i.classes)==null||o.push(s)})}setStyle(e,s){var a,i;(i=(a=this.getState(e))==null?void 0:a.styles)==null||i.push(s)}setTextStyle(e,s){var a,i;(i=(a=this.getState(e))==null?void 0:a.textStyles)==null||i.push(s)}getDirectionStatement(){return this.rootDoc.find(e=>e.stmt===Rt)}getDirection(){var e;return((e=this.getDirectionStatement())==null?void 0:e.value)??de}setDirection(e){const s=this.getDirectionStatement();s?s.value=e:this.rootDoc.unshift({stmt:Rt,value:e})}trimColon(e){return e.startsWith(":")?e.slice(1).trim():e.trim()}getData(){const e=F();return{nodes:this.nodes,edges:this.edges,other:{},config:e,direction:Kt(this.getRootDocV2())}}getConfig(){return F().state}},f(z,"StateDB"),z.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3},z),Pe=f(t=>`
defs [id$="-barbEnd"] {
    fill: ${t.transitionColor};
    stroke: ${t.transitionColor};
  }
g.stateGroup text {
  fill: ${t.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${t.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${t.stateLabelColor};
}

g.stateGroup rect {
  fill: ${t.mainBkg};
  stroke: ${t.nodeBorder};
}

g.stateGroup line {
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.transition {
  stroke: ${t.transitionColor};
  stroke-width: ${t.strokeWidth||1};
  fill: none;
}

.stateGroup .composit {
  fill: ${t.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${t.noteBorderColor};
  fill: ${t.noteBkgColor};

  text {
    fill: ${t.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${t.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${t.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${t.edgeLabelBackground};
  p {
    background-color: ${t.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${t.edgeLabelBackground};
    fill: ${t.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${t.transitionLabelColor||t.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${t.transitionLabelColor||t.tertiaryTextColor};
}

.stateLabel text {
  fill: ${t.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node .fork-join {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node circle.state-end {
  fill: ${t.innerEndBackground};
  stroke: ${t.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${t.compositeBackground||t.background};
  // stroke: ${t.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${t.stateBkg||t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}
.node polygon {
  fill: ${t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};;
  stroke-width: ${t.strokeWidth||1}px;
}
[id$="-barbEnd"] {
  fill: ${t.lineColor};
}

.statediagram-cluster rect {
  fill: ${t.compositeTitleBackground};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}

.cluster-label, .nodeLabel {
  color: ${t.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${t.stateBorder||t.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${t.compositeBackground||t.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${t.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${t.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${t.noteTextColor};
}

[id$="-dependencyStart"], [id$="-dependencyEnd"] {
  fill: ${t.lineColor};
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${t.textColor};
}

[data-look="neo"].statediagram-cluster rect {
  fill: ${t.mainBkg};
  stroke: ${t.useGradient?"url("+t.svgId+"-gradient)":t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth??1};
}
[data-look="neo"].statediagram-cluster rect.outer {
  rx: ${t.radius}px;
  ry: ${t.radius}px;
  filter: ${t.dropShadow?t.dropShadow.replace("url(#drop-shadow)",`url(${t.svgId}-drop-shadow)`):"none"}
}
`,"getStyles"),Ue=Pe;export{Me as S,Ye as a,Ve as b,Ue as s};
