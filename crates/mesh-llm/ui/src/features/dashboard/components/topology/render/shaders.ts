export function createShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export function createProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string,
) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

const LINE_SHADER_DERIVATIVE_PREAMBLE = `#extension GL_OES_standard_derivatives : enable
#define TOPOLOGY_LINE_STANDARD_DERIVATIVES 1
`;

export function buildLineFragmentShaderSource(
  fragmentSource: string,
  { useStandardDerivatives }: { useStandardDerivatives: boolean },
) {
  return useStandardDerivatives
    ? `${LINE_SHADER_DERIVATIVE_PREAMBLE}${fragmentSource}`
    : fragmentSource;
}

export const POINT_VERTEX_SHADER = `
attribute vec2 a_position;
attribute float a_size;
attribute vec4 a_color;
attribute float a_pulse;
attribute float a_twinkle;
uniform vec2 u_resolution;
uniform float u_time;
varying vec4 v_color;
varying float v_glow;
varying float v_twinkle;

void main() {
  vec2 zeroToOne = a_position / u_resolution;
  vec2 clip = zeroToOne * 2.0 - 1.0;
  gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
  float pulse = 0.94 + 0.1 * sin(u_time * (0.55 + a_pulse * 0.18) + a_pulse * 6.2831);
  gl_PointSize = a_size * pulse * (1.0 + a_twinkle * 0.12);
  v_color = a_color;
  v_glow = 0.45 + a_pulse * 0.16;
  v_twinkle = a_twinkle;
}
`;

export const DARK_POINT_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;
varying float v_glow;
varying float v_twinkle;

void main() {
  vec2 centered = gl_PointCoord * 2.0 - 1.0;
  float distanceFromCenter = length(centered);
  if (distanceFromCenter > 1.0) {
    discard;
  }

  float glow = smoothstep(1.0, 0.18, distanceFromCenter);
  float body = 1.0 - smoothstep(0.56, 0.94, distanceFromCenter);
  float core = 1.0 - smoothstep(0.0, 0.26, distanceFromCenter);
  float rim = smoothstep(0.58, 0.74, distanceFromCenter) * (1.0 - smoothstep(0.74, 0.9, distanceFromCenter));
  float perimeterFade = 1.0 - smoothstep(0.78, 1.0, distanceFromCenter);
  vec2 twinkleCoords = centered;
  float starAxis = max(
    smoothstep(0.11, 0.0, abs(twinkleCoords.x)) * smoothstep(1.0, 0.14, abs(twinkleCoords.y)),
    smoothstep(0.11, 0.0, abs(twinkleCoords.y)) * smoothstep(1.0, 0.14, abs(twinkleCoords.x))
  );
  float starDiagonal = max(
    smoothstep(0.12, 0.0, abs(twinkleCoords.x - twinkleCoords.y)) *
      smoothstep(1.08, 0.2, abs(twinkleCoords.x + twinkleCoords.y)),
    smoothstep(0.12, 0.0, abs(twinkleCoords.x + twinkleCoords.y)) *
      smoothstep(1.08, 0.2, abs(twinkleCoords.x - twinkleCoords.y))
  );
  float starNeedle = pow(max(abs(centered.x), abs(centered.y)), 0.35);
  float starSpark = smoothstep(0.24, 0.0, distanceFromCenter) * v_twinkle;
  float starFlare = ((starAxis * 0.72 + starDiagonal * 0.32) * (1.0 - starNeedle * 0.42) + starSpark * 0.48) * v_twinkle;
  float highlightMix = core * 0.22 + rim * 0.1 + starFlare * 0.54;
  vec3 highlightColor = vec3(1.0, 0.98, 0.92);
  vec3 colorMix = mix(v_color.rgb * 0.72, highlightColor, highlightMix);
  float alpha = (glow * v_glow * 0.2 + body * 0.32 + core * 0.24 + rim * 0.08 + starFlare * 0.18) * perimeterFade;

  gl_FragColor = vec4(colorMix, alpha * v_color.a);
}
`;

export const LIGHT_POINT_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;
varying float v_glow;
varying float v_twinkle;

void main() {
  vec2 centered = gl_PointCoord * 2.0 - 1.0;
  float distanceFromCenter = length(centered);
  if (distanceFromCenter > 1.0) {
    discard;
  }

  vec2 signalDir = vec2(-0.8, -0.6);
  float body = 1.0 - smoothstep(0.8, 0.93, distanceFromCenter);
  float edgeFeather = 1.0 - smoothstep(0.86, 1.0, distanceFromCenter);
  float innerBody = 1.0 - smoothstep(0.0, 0.52, distanceFromCenter);
  float denseCore = 1.0 - smoothstep(0.0, 0.2, distanceFromCenter);
  float ring =
    smoothstep(0.52, 0.64, distanceFromCenter) *
    (1.0 - smoothstep(0.64, 0.76, distanceFromCenter));
  float signalLift =
    (1.0 - smoothstep(0.0, 0.26, length(centered + signalDir * 0.11))) *
    (0.014 + v_twinkle * 0.01);
  float directionalRim = clamp(dot(centered, signalDir) * -0.5 + 0.5, 0.0, 1.0);
  vec3 rimTone = mix(v_color.rgb, vec3(0.08, 0.1, 0.14), 0.022 + ring * 0.032);
  vec3 coreTone = mix(
    v_color.rgb,
    vec3(0.992, 0.995, 1.0),
    0.026 + denseCore * 0.082 + signalLift * 0.08
  );
  vec3 ringTone = mix(v_color.rgb, vec3(0.99, 0.994, 1.0), 0.01 + directionalRim * 0.014);
  vec3 colorMix = mix(rimTone, coreTone, 0.14 + innerBody * 0.76 + denseCore * 0.05);
  colorMix = mix(colorMix, ringTone, ring * 0.068);
  float alpha = (body * 0.52 + innerBody * 0.14 + denseCore * 0.1 + ring * 0.016) * edgeFeather;

  gl_FragColor = vec4(colorMix, min(alpha, 1.0) * v_color.a);
}
`;

export const LINE_VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec4 a_color;
attribute vec2 a_lineCoord;
uniform vec2 u_resolution;
varying vec4 v_color;
varying vec2 v_lineCoord;

void main() {
  vec2 zeroToOne = a_position / u_resolution;
  vec2 clip = zeroToOne * 2.0 - 1.0;
  gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
  v_color = a_color;
  v_lineCoord = a_lineCoord;
}
`;

export const LIGHT_LINE_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;
varying vec2 v_lineCoord;

float lineSmoothstep(float edge0, float edge1, float value) {
#ifdef TOPOLOGY_LINE_STANDARD_DERIVATIVES
  float aa = max(fwidth(value) * 0.75, 0.0001);
  edge0 -= aa;
  edge1 += aa;
#endif
  return smoothstep(edge0, edge1, value);
}

void main() {
  float edgeFeather = 1.0 - lineSmoothstep(0.78, 1.0, abs(v_lineCoord.y));
  float capFeather = lineSmoothstep(0.0, 0.08, v_lineCoord.x) * (1.0 - lineSmoothstep(0.92, 1.0, v_lineCoord.x));
  gl_FragColor = vec4(v_color.rgb, v_color.a * edgeFeather * capFeather);
}
`;

export const DARK_LINE_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;
varying vec2 v_lineCoord;

float lineSmoothstep(float edge0, float edge1, float value) {
#ifdef TOPOLOGY_LINE_STANDARD_DERIVATIVES
  float aa = max(fwidth(value) * 0.75, 0.0001);
  edge0 -= aa;
  edge1 += aa;
#endif
  return smoothstep(edge0, edge1, value);
}

vec3 dodgeLift(vec3 color, float alpha) {
  vec3 base = clamp(color, 0.0, 0.96);
  vec3 lifted = min(base / max(vec3(0.34), 1.0 - base * 0.72), vec3(1.0));
  return mix(base, lifted, 0.22 + alpha * 0.12);
}

void main() {
  float edgeFeather = 1.0 - lineSmoothstep(0.74, 1.0, abs(v_lineCoord.y));
  float capFeather = lineSmoothstep(0.0, 0.08, v_lineCoord.x) * (1.0 - lineSmoothstep(0.92, 1.0, v_lineCoord.x));
  float core = 1.0 - lineSmoothstep(0.0, 0.18, abs(v_lineCoord.y));
  vec3 glowColor = dodgeLift(v_color.rgb, v_color.a);
  vec3 coreColor = mix(glowColor, vec3(1.0, 0.985, 0.955), core * 0.74);
  gl_FragColor = vec4(coreColor, v_color.a * edgeFeather * capFeather);
}
`;
