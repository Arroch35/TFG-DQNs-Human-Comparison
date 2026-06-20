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
var n_trials_per_game = 6;   // X triplets per game
var n_clips_per_trial = 3;

/* ---------------------------------
   2) GAMES + VIDEOS
--------------------------------- */
var games_data = {
  "PongNoFrameskip-v4": {
    clips: [
      'sub_pruebas_PongNoFrameskip-v4_block1_end006100.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end007011.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end012000.mp4',
      'sub_pruebas_PongNoFrameskip-v4_block1_end015800.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end019300.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end020600.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end022100.mp4',
      'sub_pruebas_PongNoFrameskip-v4_block1_end028800.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end029511.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end041511.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end044700.mp4', 
      'sub_pruebas_PongNoFrameskip-v4_block1_end045600.mp4',
      'sub_pruebas_PongNoFrameskip-v4_block1_end045700.mp4',
      'sub_pruebas_PongNoFrameskip-v4_block1_end049600.mp4',
      'sub_pruebas_PongNoFrameskip-v4_block1_end054700.mp4' 
      
    ],

    /* PASTE HERE the gameplay demo clip for Pong */
    demo: 'sub_clipsEjemplo_PongNoFrameskip-v4_block1_start000005_end001004_frames1000_dur10s.mp4',
    
    easy_triplets: [
[0, 9, 4],
[0, 9, 5],
[6, 9, 14],
[0, 9, 8],
[6, 11, 12],
[0, 9, 12],
[0, 9, 10],
[0, 9, 7],
[6, 11, 5],
[6, 9, 10],
[6, 9, 12],
[6, 11, 2],
[0, 9, 14],
[6, 11, 13],
[0, 9, 13],
[6, 11, 3],
[0, 9, 2],
[6, 9, 13],
[6, 9, 2],
[6, 9, 5],
[0, 9, 1],
[6, 9, 3],
[0, 9, 3],
[7, 12, 10],
[7, 12, 2],
[7, 12, 0],
[7, 12, 6],
[9, 11, 6],
[12, 14, 10],
[7, 12, 11],
[7, 12, 9],
[7, 12, 4],
[7, 12, 1],
[12, 14, 2],
[7, 12, 5],
[7, 12, 8],
[7, 12, 3],
[7, 12, 13],
[12, 14, 0],
[12, 14, 6],
[9, 11, 0],
[12, 14, 1],
[12, 14, 4],
[12, 14, 11],
[12, 14, 9],
[12, 14, 8],
[12, 14, 5],
[12, 14, 3],
[12, 14, 13],
[9, 11, 1],
[9, 11, 4],
[9, 11, 13],
[9, 11, 7],
[9, 11, 2],
[9, 11, 8],
[9, 11, 12],
[9, 11, 5],
[9, 11, 10],
[9, 11, 14],
[9, 11, 3],
[7, 14, 12],
[7, 14, 2],
[7, 14, 10],
[7, 14, 6],
[7, 14, 0],
[7, 14, 1],
[7, 14, 8],
[7, 14, 11],
[7, 14, 4],
[7, 14, 3],
[7, 14, 9],
[7, 14, 5],
[7, 14, 13],
],


  medium_triplets: [
[2, 3, 0],
[2, 10, 5],
[0, 5, 12],
[0, 10, 1],
[2, 14, 10],
[6, 14, 3],
[8, 9, 3],
[0, 5, 13],
[0, 5, 10],
[4, 10, 9],
[0, 11, 7],
[8, 13, 12],
[0, 11, 10],
[9, 13, 14],
[0, 5, 7],
[0, 5, 14],
[0, 14, 13],
[9, 12, 3],
[8, 13, 3],
[0, 11, 4],
[2, 13, 0],
[5, 13, 7],
[1, 11, 12],
[1, 13, 0],
[4, 10, 1],
[4, 10, 6],
[1, 3, 10],
[6, 14, 5],
[4, 10, 11],
[6, 7, 3],
[0, 11, 1],
[2, 3, 9],
[0, 11, 12],
[10, 14, 1],
[0, 11, 14],
[5, 13, 14],
[0, 11, 8],
[0, 12, 3],
[10, 12, 6],
[6, 7, 5],
[2, 3, 6],
[0, 6, 2],
[7, 10, 13],
[7, 10, 1],
[0, 5, 2],
[1, 3, 9],
[0, 5, 3],
[7, 10, 11],
[10, 14, 13],
[4, 13, 0],
[10, 12, 9],
[4, 9, 3],
[0, 6, 3],
[1, 3, 0],
[4, 13, 12],
[4, 10, 8],
[10, 12, 11],
[1, 3, 12],
[10, 12, 1],
[0, 5, 1],
[10, 13, 6],
[0, 11, 2],
[9, 13, 3],
[5, 13, 6],
[5, 8, 0],
[0, 10, 3],
[7, 10, 9],
[0, 11, 13],
[4, 10, 5],
[10, 14, 8],
[2, 7, 1],
[2, 14, 1],
[4, 13, 7],
],

hard_triplets: [
[4, 9, 2],
[1, 10, 5],
[0, 4, 1],
[1, 13, 5],
[2, 10, 1],
[8, 14, 9],
[1, 8, 10],
[0, 8, 2],
[11, 13, 12],
[2, 4, 3],
[2, 13, 9],
[1, 11, 13],
[7, 11, 13],
[2, 5, 6],
[2, 8, 12],
[5, 13, 9],
[3, 5, 7],
[6, 12, 3],
[4, 8, 12],
[0, 7, 1],
[2, 3, 5],
[8, 13, 11],
[2, 12, 6],
[2, 13, 5],
[0, 14, 1],
[8, 9, 12],
[4, 8, 3],
[0, 12, 2],
[6, 12, 13],
[4, 12, 3],
[1, 14, 9],
[0, 12, 8],
[5, 13, 11],
[0, 10, 2],
[7, 11, 3],
[3, 14, 9],
[0, 8, 1],
[1, 11, 2],
[4, 9, 12],
[2, 13, 11],
[9, 10, 3],
[4, 13, 9],
[8, 13, 0],
[3, 5, 14],
[11, 12, 3],
[10, 12, 0],
[2, 4, 12],
[1, 8, 12],
[3, 5, 11],
[6, 7, 8],
[8, 14, 3],
[10, 13, 1],
[2, 4, 0],
[6, 8, 10],
[4, 9, 7],
[2, 4, 5],
[0, 6, 12],
[8, 9, 10],
[9, 13, 12],
[10, 12, 4],
[1, 3, 14],
[1, 2, 5],
[0, 4, 3],
[11, 13, 14],
[1, 11, 4],
[1, 3, 5],
[0, 6, 4],
[1, 2, 12],
[7, 9, 3],
[4, 11, 3],
[1, 11, 7],
[4, 10, 0],
]
  },
  




  "SpaceInvadersNoFrameskip-v4": {
    clips: [
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end000511.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end001011.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end002400.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end008300.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end008700.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end009200.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end013300.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end013500.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end019500.mp4',
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end022011.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end027200.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end040600.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end069500.mp4', 
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end070200.mp4',
      'sub_pruebas_SpaceInvadersNoFrameskip-v4_block1_end092200.mp4'
    ],

    /* PASTE HERE the gameplay demo clip for Space Invaders */
    demo: 'sub_clipsEjemplo_SpaceInvadersNoFrameskip-v4_block1_start000010_end001009_frames1000_dur10s.mp4',
    
    easy_triplets: [
[1, 7, 2],
[1, 6, 4],
[0, 3, 2],
[0, 13, 14],
[1, 8, 12],
[1, 8, 9],
[1, 8, 11],
[1, 5, 12],
[0, 1, 12],
[6, 7, 9],
[0, 13, 6],
[1, 5, 9],
[1, 8, 10],
[6, 7, 12],
[1, 5, 11],
[6, 7, 10],
[0, 1, 9],
[6, 7, 11],
[0, 13, 7],
[1, 5, 10],
[1, 7, 13],
[0, 3, 9],
[0, 1, 11],
[0, 13, 5],
[0, 3, 12],
[1, 6, 12],
[0, 3, 11],
[0, 1, 10],
[1, 6, 11],
[0, 3, 10],
[1, 6, 9],
[1, 6, 10],
[5, 7, 8],
[0, 13, 4],
[0, 8, 9],
[0, 8, 12],
[0, 8, 11],
[5, 7, 0],
[1, 7, 3],
[3, 13, 0],
[1, 7, 14],
[0, 8, 10],
[0, 13, 2],
[1, 7, 4],
[5, 7, 2],
[5, 7, 13],
[0, 13, 9],
[0, 13, 12],
[5, 7, 14],
[0, 13, 11],
[5, 7, 3],
[0, 13, 10],
[5, 7, 4],
[1, 7, 9],
[1, 7, 12],
[1, 7, 10],
[1, 7, 11],
[5, 7, 9],
[5, 7, 12],
[5, 7, 10],
[5, 7, 11],
[3, 13, 8],
[3, 13, 14],
[3, 13, 1],
[3, 13, 6],
[3, 13, 5],
[3, 13, 7],
[3, 13, 4],
[3, 13, 2],
[3, 13, 9],
[3, 13, 11],
[3, 13, 12],
[3, 13, 10],
],

medium_triplets: [
[1, 3, 12],
[0, 6, 12],
[1, 3, 9],
[7, 8, 4],
[2, 7, 10],
[1, 5, 0],
[0, 5, 12],
[1, 8, 13],
[0, 6, 11],
[9, 12, 1],
[10, 12, 5],
[9, 10, 2],
[10, 12, 3],
[3, 14, 11],
[0, 5, 9],
[10, 12, 6],
[10, 12, 14],
[0, 6, 9],
[1, 13, 4],
[9, 11, 0],
[0, 5, 11],
[1, 3, 11],
[5, 7, 1],
[3, 14, 2],
[3, 8, 9],
[1, 7, 6],
[2, 7, 11],
[3, 14, 9],
[3, 8, 11],
[0, 6, 10],
[6, 8, 11],
[9, 10, 1],
[10, 12, 13],
[0, 5, 10],
[3, 14, 10],
[1, 3, 10],
[1, 5, 8],
[8, 13, 2],
[10, 12, 7],
[3, 8, 10],
[3, 8, 12],
[0, 7, 9],
[11, 12, 2],
[6, 8, 10],
[2, 6, 11],
[1, 6, 8],
[6, 8, 12],
[9, 11, 2],
[6, 8, 9],
[2, 6, 10],
[3, 14, 12],
[9, 11, 1],
[11, 12, 14],
[10, 12, 8],
[11, 12, 3],
[1, 8, 2],
[13, 14, 11],
[11, 12, 6],
[0, 8, 3],
[13, 14, 2],
[1, 6, 0],
[11, 12, 5],
[0, 7, 12],
[2, 6, 9],
[5, 6, 8],
[5, 6, 0],
[10, 12, 0],
[13, 14, 9],
[0, 3, 1],
[11, 12, 13],
[13, 14, 10],
[1, 2, 13],
[13, 14, 12],
],

hard_triplets: [
[4, 10, 13],
[2, 8, 14],
[2, 12, 5],
[4, 7, 11],
[1, 4, 12],
[2, 6, 0],
[4, 10, 7],
[7, 14, 11],
[1, 4, 9],
[2, 12, 0],
[4, 14, 2],
[3, 8, 5],
[2, 7, 13],
[4, 14, 10],
[1, 6, 5],
[0, 5, 4],
[1, 13, 14],
[2, 8, 12],
[4, 14, 6],
[0, 4, 12],
[0, 7, 14],
[4, 8, 9],
[4, 8, 10],
[2, 4, 10],
[0, 6, 4],
[4, 10, 1],
[0, 4, 9],
[8, 13, 6],
[0, 7, 2],
[8, 13, 7],
[4, 14, 9],
[4, 8, 12],
[1, 4, 11],
[1, 3, 4],
[10, 11, 7],
[2, 12, 13],
[2, 7, 3],
[2, 7, 14],
[2, 7, 12],
[4, 8, 11],
[8, 14, 11],
[2, 6, 13],
[8, 13, 14],
[5, 13, 9],
[2, 4, 9],
[5, 14, 12],
[10, 12, 4],
[11, 12, 10],
[5, 13, 11],
[6, 13, 4],
[6, 8, 14],
[2, 12, 14],
[4, 14, 12],
[5, 13, 12],
[0, 4, 10],
[0, 14, 2],
[5, 13, 10],
[3, 14, 6],
[7, 13, 12],
[9, 10, 3],
[2, 6, 3],
[0, 7, 4],
[2, 4, 11],
[2, 5, 10],
[3, 8, 4],
[9, 10, 14],
[3, 6, 11],
[2, 5, 11],
[0, 4, 11],
[3, 14, 5],
[10, 11, 0],
[7, 13, 10],
]
  },
  
  "MsPacmanNoFrameskip-v4": {
    clips: [
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end001900.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end002550.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end003500.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end005700.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end006700.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end007950.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end011200.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end011350.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end015750.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end016750.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end023100.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end024500.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end025000.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end028700.mp4',
      'sub_pruebas_MsPacmanNoFrameskip-v4_block1_end032400.mp4'
    ],

    /* PASTE HERE the gameplay demo clip for Ms. Pac-Man */
    demo: 'sub_clipsEjemplo_MsPacmanNoFrameskip-v4_block1_start000200_end001365_frames1166_dur10s.mp4',
    
    easy_triplets: [
[0, 3, 6],
[12, 14, 0],
[2, 10, 12],
[1, 10, 12],
[12, 13, 14],
[12, 14, 8],
[3, 4, 8],
[0, 3, 9],
[11, 12, 8],
[0, 3, 11],
[5, 8, 12],
[5, 10, 14],
[13, 14, 0],
[11, 12, 10],
[13, 14, 1],
[5, 10, 4],
[0, 3, 7],
[12, 14, 10],
[3, 4, 10],
[2, 10, 13],
[0, 3, 5],
[0, 3, 8],
[1, 10, 13],
[0, 3, 10],
[0, 1, 13],
[5, 8, 13],
[0, 3, 2],
[0, 1, 12],
[12, 13, 11],
[0, 3, 13],
[13, 14, 8],
[5, 10, 12],
[0, 3, 14],
[13, 14, 10],
[5, 10, 13],
[0, 3, 12],
[12, 13, 4],
[12, 13, 9],
[12, 13, 3],
[12, 13, 6],
[8, 10, 5],
[12, 13, 2],
[6, 7, 11],
[12, 13, 7],
[12, 13, 5],
[6, 7, 4],
[12, 13, 0],
[12, 13, 1],
[8, 10, 2],
[6, 7, 14],
[12, 13, 8],
[6, 7, 1],
[8, 10, 1],
[6, 7, 9],
[6, 7, 0],
[6, 7, 10],
[6, 7, 5],
[12, 13, 10],
[8, 10, 0],
[6, 7, 3],
[6, 7, 2],
[6, 7, 8],
[8, 10, 7],
[6, 7, 12],
[8, 10, 6],
[6, 7, 13],
[8, 10, 9],
[8, 10, 3],
[8, 10, 11],
[8, 10, 14],
[8, 10, 4],
[8, 10, 12],
[8, 10, 13],
],

medium_triplets: [
[3, 4, 11],
[0, 1, 7],
[2, 5, 11],
[11, 13, 0],
[3, 4, 13],
[11, 14, 3],
[9, 12, 8],
[3, 4, 9],
[11, 12, 9],
[4, 11, 2],
[2, 10, 7],
[7, 10, 12],
[1, 10, 9],
[4, 12, 8],
[6, 11, 0],
[2, 5, 4],
[4, 6, 13],
[6, 14, 8],
[3, 13, 1],
[2, 8, 4],
[7, 9, 0],
[2, 8, 14],
[11, 13, 5],
[2, 10, 6],
[13, 14, 9],
[0, 4, 2],
[1, 6, 13],
[9, 13, 10],
[0, 8, 12],
[0, 4, 5],
[0, 10, 14],
[4, 11, 5],
[11, 14, 2],
[11, 14, 0],
[9, 14, 8],
[0, 8, 13],
[1, 8, 14],
[11, 14, 5],
[3, 4, 14],
[12, 14, 7],
[6, 11, 1],
[6, 10, 13],
[0, 1, 9],
[4, 6, 5],
[6, 11, 13],
[4, 14, 2],
[0, 1, 5],
[1, 10, 11],
[1, 7, 12],
[11, 13, 1],
[9, 12, 10],
[5, 8, 0],
[4, 14, 1],
[7, 9, 3],
[1, 10, 3],
[9, 14, 10],
[7, 9, 5],
[11, 14, 1],
[4, 9, 8],
[4, 6, 2],
[0, 4, 10],
[6, 11, 3],
[3, 13, 8],
[0, 4, 13],
[2, 5, 12],
[4, 12, 10],
[4, 14, 5],
[5, 10, 0],
[11, 12, 7],
[2, 8, 12],
[0, 4, 8],
[4, 13, 8],
[7, 9, 12],
],

hard_triplets: [
[0, 8, 11],
[1, 8, 7],
[4, 11, 9],
[5, 6, 12],
[6, 13, 8],
[6, 10, 3],
[3, 12, 8],
[1, 3, 12],
[9, 14, 6],
[1, 8, 6],
[4, 7, 5],
[0, 9, 13],
[0, 6, 8],
[7, 11, 1],
[9, 13, 5],
[1, 5, 14],
[0, 6, 9],
[0, 4, 11],
[1, 6, 9],
[7, 10, 3],
[11, 14, 9],
[9, 12, 6],
[12, 14, 4],
[7, 14, 1],
[7, 14, 5],
[7, 8, 12],
[6, 14, 1],
[4, 7, 12],
[4, 12, 2],
[6, 14, 5],
[1, 10, 7],
[6, 12, 8],
[4, 9, 1],
[4, 7, 13],
[6, 11, 14],
[1, 10, 6],
[0, 4, 9],
[6, 14, 0],
[5, 10, 2],
[5, 6, 13],
[0, 2, 13],
[7, 9, 14],
[0, 10, 9],
[2, 10, 1],
[0, 10, 11],
[5, 8, 2],
[0, 1, 10],
[1, 7, 3],
[13, 14, 11],
[2, 5, 1],
[7, 9, 4],
[9, 14, 3],
[1, 9, 13],
[12, 14, 9],
[3, 9, 10],
[1, 8, 9],
[6, 14, 3],
[4, 12, 5],
[7, 14, 2],
[3, 13, 6],
[4, 12, 1],
[5, 7, 12],
[9, 12, 2],
[2, 8, 7],
[0, 2, 12],
[4, 13, 2],
[0, 4, 7],
[2, 5, 7],
[1, 2, 12],
[3, 12, 10],
[6, 14, 2],
[7, 11, 5],
]
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

function sampleBalancedTriplets(game_name) {

  // ---------------------------------------------
  // 1) FIXED BALANCE (2 per difficulty)
  // ---------------------------------------------
  var difficulty_pool = [
    "easy_triplets",
    "easy_triplets",
    "medium_triplets",
    "medium_triplets",
    "hard_triplets",
    "hard_triplets"
  ];

  // ---------------------------------------------
  // 2) SHUFFLE TRIAL ORDER (THIS FIXES YOUR ISSUE)
  // ---------------------------------------------
  difficulty_pool = shuffleArray(difficulty_pool);

  var used_indices = {
    easy_triplets: new Set(),
    medium_triplets: new Set(),
    hard_triplets: new Set()
  };

  var results = [];

  for (var i = 0; i < difficulty_pool.length; i++) {

    var difficulty = difficulty_pool[i];
    var pool = games_data[game_name][difficulty];

    var triplet = null;
    var idx = null;

    // try avoid repetition inside same difficulty 
    for (var attempt = 0; attempt < 20; attempt++) {

      idx = Math.floor(Math.random() * pool.length);

      if (!used_indices[difficulty].has(idx)) {
        used_indices[difficulty].add(idx);
        triplet = pool[idx];
        break;
      }
    }

    // fallback if needed
    if (triplet === null) {
      idx = Math.floor(Math.random() * pool.length);
      triplet = pool[idx];
    }

    var real_triplet = triplet.map(function(idx) {
      return games_data[game_name].clips[idx];
    });
    
    results.push({
      difficulty: difficulty,
      triplet: shuffleArray(real_triplet)
    });
  }

  return results;
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
  trial_duration: 0
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
function createTripletTrial(triplet, game_name, trial_in_game, total_trials_in_game, difficulty) {
  console.log(difficulty);
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
      difficulty: difficulty,
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

      //Esta funcion está hecha así porque con 4 frames el browser lo jodia y ponia solo 3
      document.querySelectorAll(".play-btn").forEach(function(btn) {
        btn.addEventListener("click", function() {
          var idx = parseInt(this.dataset.vid);
          var v = vids[idx];
          
          // 1. Reset
          v.pause();
          v.currentTime = 0;
      
          // 2. Define the exact timestamps for 2fps (4 frames)
          // Frame 1: 0.0s, Frame 2: 0.5s, Frame 3: 1.0s, Frame 4: 1.5s
          var frameTimes = [0.0, 0.5, 1.0, 1.5];
          var currentF = 0;
      
          // 3. Manually step through the frames every 500ms
          var timer = setInterval(function() {
            currentF++;
            if (currentF < frameTimes.length) {
              v.currentTime = frameTimes[currentF];
            } else {
              clearInterval(timer);
              v.pause();
            }
          }, 500); // 500ms = 2fps
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
          difficulty: difficulty,
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

  var game_triplets = sampleBalancedTriplets(game_name);

  timeline.push(
    createGameStartScreen(
      game_name,
      game_index + 1,
      game_order.length,
      game_demo
    )
  );

  game_triplets.forEach(function(item, idx) {
  
  
    var triplet = item.triplet;
    var difficulty = item.difficulty;
    
  
    timeline.push(blank_screen);
  
    timeline.push(createTripletTrial(
      triplet,
      game_name,
      idx + 1,
      game_triplets.length,
      difficulty
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
      <p>Antes de terminar, por favor, responde unas preguntas sobre tu experiencia.</p>
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
      prompt: "¿Te han parecido difíciles de interpretar los clips?",
      labels: ["Nada", "Poco", "Neutral", "Bastante", "Mucho"],
      required: true
    },
    {
      prompt: "¿Qué tan seguro/a te sentías al elegir el clip más diferente?",
      labels: ["Nada seguro/a", "Poco seguro/a", "Neutral", "Bastante seguro/a", "Muy seguro/a"],
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
    data.clip_interpretation_difficulty = responses.Q2;
    data.decision_confidence = responses.Q3;
    data.perceived_experiment_length = responses.Q4;
    
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