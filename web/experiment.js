/* initialize jsPsych */
var jsPsych = initJsPsych({
  on_finish: function() {
    console.log("Experiment finished.");
  }
});

/* create timeline */
var timeline = [];

/* ---------------------------------
   1) CONFIGURATION
--------------------------------- */
var n_trials_per_game = 10;   // X triplets per game
var n_clips_per_trial = 3;

/* ---------------------------------
   2) GAMES + VIDEOS
--------------------------------- */
var games_data = {
  "PongNoFrameskip-v4": {
    clips: [
      'sub_pruebas_PongNoFrameskip-v4_block1_end006100_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end007011_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end012000_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end014511_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end015011_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end015800_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end019300_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end028511_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end045700_frames12.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end054700_frames12.mp4'
    ],

    /* PASTE HERE the gameplay demo clip for Pong */
    demo: 'sub_clipsEjemplo_PongNoFrameskip-v4_block1_start000005_end001004_frames1000_dur10s.mp4'
  },

  "SpaceInvadersNoFrameskip-v4": {
    clips: [
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end000511_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end002400_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end005011_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end008300_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end011700_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end013700_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end018500_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end019500_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end037100_frames12.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end068400_frames12.mp4'
    ],

    /* PASTE HERE the gameplay demo clip for Space Invaders */
    demo: 'sub_clipsEjemplo_SpaceInvadersNoFrameskip-v4_block1_start000010_end001009_frames1000_dur10s.mp4'
  },
  
  "MsPacmanNoFrameskip-v4": {
    clips: [
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end001900_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end006700_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end007150_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end008900_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end011350_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end015850_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end020850_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end025000_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end028700_frames12.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end032400_frames12.mp4'
    ],

    /* PASTE HERE the gameplay demo clip for Ms. Pac-Man */
    demo: 'sub_clipsEjemplo_MsPacmanNoFrameskip-v4_block1_start000200_end001365_frames1166_dur10s.mp4'
  }
};

var game_order = shuffleArray([
  "MsPacmanNoFrameskip-v4",
  "PongNoFrameskip-v4",
  "SpaceInvadersNoFrameskip-v4"
]);

/* flatten all videos for preload */
var all_videos = [];
game_order.forEach(function(game) {
  all_videos = all_videos.concat(games_data[game].clips);
  all_videos.push(games_data[game].demo); // also preload gameplay demo clips
});

console.log("All video paths:", all_videos);

/* ---------------------------------
   3) HELPERS
--------------------------------- */
function shuffleArray(array) {
  var arr = array.slice();
  for (var i = arr.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
  return arr;
}

function gameLabel(gameName) {
  if (gameName.includes("MsPacman")) return "Ms. Pac-Man";
  if (gameName.includes("Pong")) return "Pong";
  if (gameName.includes("SpaceInvaders")) return "Space Invaders";
  return gameName;
}

function generateUniqueTriplets(video_list, n_trials, n_clips_per_trial) {
  var trials = [];
  var used_triplets = new Set();

  var max_possible = combinationCount(video_list.length, n_clips_per_trial);
  if (n_trials > max_possible) {
    console.warn("Requested", n_trials, "triplets but only", max_possible, "unique triplets possible.");
    n_trials = max_possible;
  }

  while (trials.length < n_trials) {
    var triplet = shuffleArray(video_list).slice(0, n_clips_per_trial);
    var key = triplet.slice().sort().join("|");
    if (!used_triplets.has(key)) {
      trials.push(triplet);
      used_triplets.add(key);
    }
  }

  return trials;
}

function combinationCount(n, k) {
  if (k > n) return 0;
  var result = 1;
  for (var i = 1; i <= k; i++) {
    result = result * (n - i + 1) / i;
  }
  return Math.round(result);
}

function getTripletClassification(triplet, chosen_clip) {
  var similar = triplet.filter(function(c) {
    return c !== chosen_clip;
  });

  return {
    similar_clip_1: similar[0],
    similar_clip_2: similar[1],
    odd_clip: chosen_clip
  };
}

/* ---------------------------------
   4) PRELOAD ALL VIDEOS
--------------------------------- */
var preload = {
  type: jsPsychPreload,
  video: all_videos
};
timeline.push(preload);

/* ---------------------------------
   5) INTRO SCREEN
--------------------------------- */
var intro = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div style="max-width:900px; margin:auto; font-size:16px; line-height:1.6; text-align:justify;">

      <h2 style="font-size:32px; text-align:center; margin-bottom:20px;">Instrucciones</h2>
    
      <p>A continuación verás varios <strong>tríos de clips</strong> cortos de videojuegos.</p>
    
      <p>En cada prueba aparecerán <strong>3 clips del mismo juego</strong>.</p>
    
      <p>Tu tarea es <strong>elegir el clip que sea más diferente de los otros dos</strong>.</p>
    
      <p><strong>¿Qué significa “más diferente”?</strong><br>
      Ponte en <strong>la piel del jugador</strong> y fíjate en <strong>lo que está ocurriendo en el juego</strong>.</p>
    
      <p>Debes elegir el clip que muestre una <strong>situación de juego distinta</strong> a la de los otros dos.</p>
     
        <p>
          Para decidirlo, puedes fijarte en:
        </p>
        
        <p style=" margin-left:10px;">
          • qué está ocurriendo<br>
          • cómo se relacionan entre sí los elementos del juego<br>
          • cómo se están moviendo<br>
          • si la situación del jugador es parecida o diferente
        </p>
    
      <p><strong>Importante:</strong><br>
      No te fijes solo en si un personaje u objeto está en otra parte de la pantalla.<br>
      Dos clips pueden ser parecidos aunque algunos elementos estén colocados de forma diferente, <strong>si la situación del juego es esencialmente la misma</strong>.</p>
    
      <p>Intenta no basarte en:</p>
    
      <p style=" margin-left:10px;">
        • marcador, puntos o vidas<br>
        • números o texto en pantalla<br>
        • colores o brillo<br>
        • pequeños cambios visuales
      </p>
    
      <p>Lo importante es <strong>la situación de juego</strong>, no tanto cómo se ve.</p>
    
      <p>Los clips están <strong>ralentizados</strong> para que te sea más fácil entender cada situación.</p>
    
      <p><strong>Recuerda:</strong> no hay respuestas correctas ni incorrectas. Elige la opción que te parezca más adecuada según la <strong>situación del juego</strong>.</p>
    
    </div>
  `,
  choices: ["Empezar"]
};
timeline.push(intro);

/* ---------------------------------
   6) BLANK SCREEN BETWEEN TRIALS
--------------------------------- */
var blank_screen = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '',
  choices: "NO_KEYS",
  trial_duration: 100
};

/* ---------------------------------
   7) GAME TRANSITION SCREEN WITH DEMO
--------------------------------- */
var playerDescription = {
  "PongNoFrameskip-v4": "Imagina que controlas la paleta de la derecha. Tu objetivo es devolver la pelota para que el oponente no la alcance.",
  "SpaceInvadersNoFrameskip-v4": "Imagina que controlas la nave espacial. Debes disparar a los enemigos, esquivar sus disparos y evitar que lleguen a la base.",
  "MsPacmanNoFrameskip-v4": "Imagina que controlas a Ms. Pac-Man. Tu objetivo es comer todas las píldoras mientras evitas a los fantasmas. Al comer las píldoras grandes, puedes comer a los fantasmas mientras están azules."
};

function createGameStartScreen(gameName, blockIndex, totalBlocks, demoClip) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <style>
        .game-start-container {
          max-width: 1000px;
          margin: auto;
          text-align: center;
          font-family: sans-serif;
        }

        .game-start-title {
          font-size: 34px;
          margin-bottom: 12px;
        }

        .game-start-subtitle {
          font-size: 24px;
          margin-bottom: 24px;
          line-height: 1.6;
        }

        .demo-video-wrapper {
          margin: 30px auto;
          width: 260px;
          max-width: 80vw;
          aspect-ratio: 160 / 210;
          background: black;
          border-radius: 16px;
          overflow: hidden;
          box-shadow: 0 8px 22px rgba(0,0,0,0.15);
        }

        .demo-video-wrapper video {
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
          background: black;
        }

        .demo-caption {
          font-size: 20px;
          color: #555;
          margin-top: 14px;
          margin-bottom: 24px;
        }

        @media (max-width: 1000px) {
          .game-start-title {
            font-size: 28px;
          }

          .game-start-subtitle {
            font-size: 20px;
          }

          .demo-caption {
            font-size: 17px;
          }
        }
      </style>

      <div class="game-start-container">
        <div class="game-start-title">Bloque ${blockIndex} de ${totalBlocks}</div>
        <div class="game-start-subtitle">
          Ahora comenzarás el bloque del juego:<br>
          <strong style="font-size:38px;">${gameLabel(gameName)}</strong>
        </div>
        
        <div class="player-info" style="font-size:16px; margin-bottom:12px; color:#333;">
          ${playerDescription[gameName]}
        </div>
        
        <div class="demo-caption">
          Puedes ver este clip para entender cómo es el juego.
        </div>

        <div class="demo-video-wrapper">
          <video id="demo-video" playsinline muted preload="auto" controls>
            <source src="${demoClip}" type="video/mp4">
          </video>
        </div>


        <p style="font-size:22px;">Pulsa <strong>Continuar</strong> cuando estés listo/a.</p>
      </div>
    `,
    choices: ["Continuar"],
    data: {
      task: "game_start",
      game_name: gameName,
      block_index: blockIndex,
      demo_clip: demoClip
    },
    on_load: function() {
      var demoVideo = document.getElementById("demo-video");
      if (demoVideo) {
        demoVideo.currentTime = 0;
      }
    }
  };
}

/* ---------------------------------
   8) TRIAL TEMPLATE
--------------------------------- */
function createTripletTrial(triplet, game_name, trial_in_game, total_trials_in_game) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: function() {
      var video1 = triplet[0], video2 = triplet[1], video3 = triplet[2];
      return `
        <style>
          #rotate-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: white;
            z-index: 99999;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 30px;
            font-family: sans-serif;
          }

          #rotate-overlay-content {
            max-width: 500px;
          }

          #rotate-overlay h2 {
            font-size: 32px;
            margin-bottom: 15px;
          }

          #rotate-overlay p {
            font-size: 22px;
            line-height: 1.5;
          }

          .triplet-container {
            text-align: center;
            max-width: 1400px;
            margin: auto;
          }

          .trial-title {
            font-size: 32px;
            margin-bottom: 8px;
          }

          .trial-subtitle {
            font-size: 20px;
            color: #555;
            margin-bottom: 20px;
          }

          .top-controls {
            margin-bottom: 24px;
          }

          .clips-row {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 20px;
            flex-wrap: nowrap;
          }

          .clip-box {
            width: 30%;
            min-width: 220px;
            border: 4px solid transparent;
            border-radius: 18px;
            padding: 10px;
            transition: 0.15s ease;
            background: #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            box-sizing: border-box;
            overflow: hidden;
          }

          .clip-box.selected {
            border-color: #2b7cff;
            background: #e0e0e0;
            box-shadow: 0 6px 18px rgba(43,124,255,0.18);
          }

          .video-frame {
            position: relative;
            width: 100%;
            aspect-ratio: 160 / 210;
            background: black;
            border-radius: 12px;
            overflow: hidden;
          }

          .video-frame video {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            background: black;
          }

          .clip-controls {
            margin-top: 14px;
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
          }

          .select-btn {
            margin-top: 14px;
          }

          .btn {
            border: none;
            border-radius: 12px;
            padding: 14px 20px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            transition: 0.15s ease;
            min-height: 52px;
          }

          .btn:hover {
            transform: translateY(-1px);
            opacity: 0.96;
          }

          .btn-play {
            background: #e9f3ff;
            color: #1456a0;
            min-width: 70px;
            font-size: 24px;
          }

          .btn-select {
            background: #eaf8ee;
            color: #1f7a39;
            min-width: 180px;
          }

          .btn-all {
            background: #f1ebff;
            color: #5b33a8;
            font-size: 20px;
            padding: 16px 26px;
            min-width: 240px;
          }

          .btn-continue {
            background: #2b7cff;
            color: white;
            font-size: 20px;
            padding: 16px 32px;
            border-radius: 14px;
            min-width: 220px;
          }

          .btn-continue:disabled {
            background: #b8c7dd;
            color: #eef3fa;
            cursor: not-allowed;
            transform: none;
          }

          @media (max-width: 1000px) {
            .clips-row {
              gap: 10px;
            }

            .clip-box {
              width: 32%;
              min-width: 100px;
              max-width: 220px;
              padding: 6px;
            }

            .btn {
              font-size: 15px;
              padding: 10px 12px;
              min-height: 46px;
            }

            .btn-play {
              font-size: 20px;
              min-width: 56px;
            }

            .btn-select {
              min-width: 130px;
            }

            .btn-all {
              font-size: 16px;
              min-width: 180px;
              padding: 12px 18px;
            }

            .btn-continue {
              font-size: 18px;
              min-width: 180px;
            }
          }
        </style>

        <div id="rotate-overlay">
          <div id="rotate-overlay-content">
            <h2>Por favor, gira tu móvil</h2>
            <p>Este experimento debe completarse en <strong>modo horizontal</strong>.</p>
            <p>Gira el teléfono para continuar.</p>
          </div>
        </div>

        <div class="triplet-container" id="main-trial-content">
          <div class="trial-title">¿Qué clip muestra la situación más diferente?</div>
          <div class="trial-subtitle">
            ${gameLabel(game_name)} · Trío ${trial_in_game} de ${total_trials_in_game}
          </div>

          <div class="top-controls">
            <button id="play-all-btn" class="btn btn-all">▶ Reproducir los 3</button>
          </div>

          <div class="clips-row">

            <div class="clip-box" id="clip-box-0">
              <div class="video-frame">
                <video id="vid0" playsinline preload="auto">
                  <source src="${video1}" type="video/mp4">
                </video>
              </div>
              <div class="clip-controls">
                <button class="btn btn-play play-btn" data-vid="0">▶</button>
              </div>
              <div class="select-btn">
                <button class="btn btn-select choose-btn" data-choice="0">Seleccionar</button>
              </div>
            </div>

            <div class="clip-box" id="clip-box-1">
              <div class="video-frame">
                <video id="vid1" playsinline preload="auto">
                  <source src="${video2}" type="video/mp4">
                </video>
              </div>
              <div class="clip-controls">
                <button class="btn btn-play play-btn" data-vid="1">▶</button>
              </div>
              <div class="select-btn">
                <button class="btn btn-select choose-btn" data-choice="1">Seleccionar</button>
              </div>
            </div>

            <div class="clip-box" id="clip-box-2">
              <div class="video-frame">
                <video id="vid2" playsinline preload="auto">
                  <source src="${video3}" type="video/mp4">
                </video>
              </div>
              <div class="clip-controls">
                <button class="btn btn-play play-btn" data-vid="2">▶</button>
              </div>
              <div class="select-btn">
                <button class="btn btn-select choose-btn" data-choice="2">Seleccionar</button>
              </div>
            </div>

          </div>

          <div style="margin-top:32px;">
            <button id="continue-btn" class="btn btn-continue" disabled>Continuar</button>
          </div>
        </div>
      `;
    },
    choices: [],
    data: {
      task: "odd_one_out",
      game_name: game_name,
      clip_1: triplet[0],
      clip_2: triplet[1],
      clip_3: triplet[2],
      trial_in_game: trial_in_game,
      total_trials_in_game: total_trials_in_game
    },
    on_load: function() {
      var startTime = performance.now();
      var selectedChoice = null;

      var vids = [
        document.getElementById("vid0"),
        document.getElementById("vid1"),
        document.getElementById("vid2")
      ];

      var boxes = [
        document.getElementById("clip-box-0"),
        document.getElementById("clip-box-1"),
        document.getElementById("clip-box-2")
      ];

      var continueBtn = document.getElementById("continue-btn");
      var playAllBtn = document.getElementById("play-all-btn");

      function isMobileDevice() {
        return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
      }

      function checkOrientation() {
        var overlay = document.getElementById("rotate-overlay");
        var mainContent = document.getElementById("main-trial-content");

        if (isMobileDevice() && window.innerHeight > window.innerWidth) {
          overlay.style.display = "flex";
          mainContent.style.visibility = "hidden";
        } else {
          overlay.style.display = "none";
          mainContent.style.visibility = "visible";
        }
      }

      checkOrientation();
      window.addEventListener("resize", checkOrientation);
      window.addEventListener("orientationchange", checkOrientation);

      document.querySelectorAll(".play-btn").forEach(function(btn) {
        btn.addEventListener("click", function() {
          var idx = parseInt(this.dataset.vid);
          vids[idx].pause();
          vids[idx].currentTime = 0;
          vids[idx].play().catch(function(e) {
            console.log(e);
          });
        });
      });

      document.querySelectorAll(".choose-btn").forEach(function(btn) {
        btn.addEventListener("click", function() {
          selectedChoice = parseInt(this.dataset.choice);
          boxes.forEach(function(b) { b.classList.remove("selected"); });
          boxes[selectedChoice].classList.add("selected");
          continueBtn.disabled = false;
        });
      });

      playAllBtn.addEventListener("click", function() {
        vids.forEach(function(v) {
          v.pause();
          v.currentTime = 0;
        });
        vids.forEach(function(v) {
          v.play().catch(function(e) {
            console.log(e);
          });
        });
      });

      continueBtn.addEventListener("click", function() {
        if (selectedChoice === null) return;
      
        var rt = performance.now() - startTime;
        var chosen_clip = triplet[selectedChoice];
        var classified = getTripletClassification(triplet, chosen_clip);
      
        window.removeEventListener("resize", checkOrientation);
        window.removeEventListener("orientationchange", checkOrientation);
      
        vids.forEach(function(v) {
          v.pause();
          v.currentTime = 0;
        });
      
        jsPsych.finishTrial({
          task: "odd_one_out",
          game_name: game_name,
          clip_1: triplet[0],
          clip_2: triplet[1],
          clip_3: triplet[2],
          chosen_position: selectedChoice,
          chosen_clip: chosen_clip,
      
          // NEW: directly save the triplet in the exact format you want
          similar_clip_1: classified.similar_clip_1,
          similar_clip_2: classified.similar_clip_2,
          odd_clip: classified.odd_clip,
      
          rt: rt,
          trial_in_game: trial_in_game,
          total_trials_in_game: total_trials_in_game,
          screen_w: window.innerWidth,
          screen_h: window.innerHeight,
          is_mobile: isMobileDevice()
        });
      });
    }
  };
}

/* ---------------------------------
   9) BUILD TIMELINE BY GAME BLOCKS
--------------------------------- */
game_order.forEach(function(game_name, game_index) {
  var game_videos = games_data[game_name].clips;
  var game_demo = games_data[game_name].demo;

  var game_triplets = generateUniqueTriplets(
    game_videos,
    n_trials_per_game,
    n_clips_per_trial
  );

  timeline.push(
    createGameStartScreen(
      game_name,
      game_index + 1,
      game_order.length,
      game_demo
    )
  );

  game_triplets.forEach(function(triplet, idx) {
    timeline.push(blank_screen);
    timeline.push(createTripletTrial(
      triplet,
      game_name,
      idx + 1,
      game_triplets.length
    ));
  });
});

/* ---------------------------------
   10) POST-EXPERIMENT SURVEY
--------------------------------- */
var feedback_survey_1 = {
  type: jsPsychSurveyLikert,
  preamble: `
    <div style="max-width:900px; margin:auto; font-size:24px; line-height:1.6;">
      <h2>Cuestionario breve</h2>
      <p>Antes de terminar, responde unas preguntas sobre tu experiencia.</p>
    </div>
  `,
  questions: [
    {
      prompt: "¿Juegas habitualmente a videojuegos?",
      labels: ["Nunca", "Rara vez", "A veces", "Frecuentemente", "Muy frecuentemente"],
      required: true
    },
    {
      prompt: "¿Conocías alguno de estos juegos antes del experimento?",
      labels: ["No, ninguno", "Me sonaba alguno", "Conocía alguno", "Conocía varios", "Conocía todos"],
      required: true
    },
    {
      prompt: "¿La tarea te ha parecido clara?",
      labels: ["Nada clara", "Poco clara", "Neutral", "Bastante clara", "Muy clara"],
      required: true
    },
    {
      prompt: "¿Te resultó difícil decidir cuál era el clip más diferente?",
      labels: ["Nada difícil", "Poco difícil", "Neutral", "Bastante difícil", "Muy difícil"],
      required: true
    },
    {
      prompt: "¿Te han parecido difíciles de interpretar los clips?",
      labels: ["Nada", "Poco", "Neutral", "Bastante", "Mucho"],
      required: true
    },
    {
      prompt: "¿Qué tan largo te ha parecido el experimento?",
      labels: [
        "Muy largo",
        "Un poco largo",
        "Adecuado",
        "Podría haber sido un poco más largo",
        "Podría hacer más intentos"
      ],
      required: true
    },
    {
      prompt: "¿Qué tan seguro/a te sentías al elegir el clip más diferente?",
      labels: ["Nada seguro/a", "Poco seguro/a", "Neutral", "Bastante seguro/a", "Muy seguro/a"],
      required: true
    }
  ],
  button_label: "Acabar",
  data: {
    task: "post_survey_likert"
  },
  on_finish: function(data) {
    var responses = data.response || {};
    data.video_game_frequency = responses.Q0;
    data.prior_game_familiarity = responses.Q1;
    data.task_clarity = responses.Q2;
    data.odd_selection_difficulty = responses.Q3;
    data.clip_interpretation_difficulty = responses.Q4;
    data.perceived_experiment_length = responses.Q5;
    data.decision_confidence = responses.Q6;
  }
};


timeline.push(feedback_survey_1);

/* ---------------------------------
   11) END SCREEN
--------------------------------- */
var end_screen = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div style="max-width:900px; margin:auto; font-size:28px; line-height:1.8; text-align:center;">
      <h2>¡Muchas gracias por participar!</h2>
      <p>Has completado el experimento correctamente.</p>
      <p>Tu participación es de gran ayuda para esta investigación.</p>
      <p style="margin-top:30px; font-size:22px; color:#666;">
        Puedes cerrar esta ventana cuando quieras.
      </p>
    </div>
  `,
  choices: "NO_KEYS",
  trial_duration: 5000
};

timeline.push(end_screen);

/* start the experiment */
jsPsych.run(timeline);

// todo: HACER LO ULTIMO QUE ME HA DICHE+O EL CHAT. TENGO QUE HACER ESO + QUE SE HAGA POR TODOS LOS JUEGOS, MAS QUE SE RECOGAN LOS DATOS COMO YO QUIERO
// LUEGO FALTARÁ ENQUESTA PARA MEJORAR EL JUEGO Y PROBAR EL TIEMPO QUE SE PUEDE TARDAR SIENDO REALISTAS. pROBAR TAMBIEN ESTO EN EL MOVIL

//TODO: Falta: -QUE En las pantallas de trasnicion entre juego y juego se puedan ver los clips de gameplay
//             - Que se haga la enquesta de satisfacción
//             - Que se guarden los datos correctamente
//             - Probar que funcione TODO!!!

// Se me guarrdan raro las cosas, y al finalizar no se tiene que poner de pulsar a una tecle, sino que sea un click o un tab, o que ni haga falta, porque mucha gente saldrá de ahí sin clicar
// Hacer mejor la survey, porque ahora son dos y eso no me gusta, y ver si lo de la pantalla en blanco ayuda o no

// En la encuesta preguntar si conocian los juegos y cuales. de ahí preguntar si le ha sido facil entender le juego y tal

//FALTA REVISAR QUE ESTÉ TODO BIEN. MEJORAR LAS INTRUCCIONES Y PROBAR EL EXPERIMENTO POR EL TIEMPO
// INSTRUCIONES DECIR QUE NO SE FIJE EN LA PUNTUACIÓN Y ESO, PERO PREGUNTAR AL CHAT ANTES SI ES BIEN IDEA