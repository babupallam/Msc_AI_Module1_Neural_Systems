% Non-Linearly Separable (NLS) Dataset 1
P1 = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
       2 1 4 3 6 5 8 7 10 9 11 13 12 14 16 15 18 17 20 19 ];
T1 = [ 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 ];

% Non-Linearly Separable (NLS) Dataset 2
P2 = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
       3 4 3 4 3 4 3 4 5 5 6 7 6 7 6 7 8 9 8 9 ];
T2 = [ 0 0 1 1 0 0 1 1 0 1 1 0 0 1 1 0 0 1 1 0 ];

% Non-Linearly Separable (NLS) Dataset 3
P3 = [ 1 3 2 5 4 7 6 9 8 11 10 13 12 15 14 17 16 19 18 20;
       2 4 6 8 10 12 14 16 18 20 1 3 5 7 9 11 13 15 17 19 ];
T3 = [ 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 ];

% Non-Linearly Separable (NLS) Dataset 4
P4 = [ 1 5 2 6 3 7 4 8 9 10 11 12 13 14 15 16 17 18 19 20;
       10 9 8 7 6 5 4 3 2 1 11 13 12 14 16 15 18 17 20 19 ];
T4 = [ 0 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1 0 ];

% Non-Linearly Separable (NLS) Dataset 5
P5 = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
       1 3 2 4 3 5 4 6 5 7 6 8 7 9 8 10 9 11 10 12 ];
T5 = [ 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 ];

% Display datasets
figure;
subplot(3, 2, 1); plotpv(P1, T1); title('NLS Dataset 1');
subplot(3, 2, 2); plotpv(P2, T2); title('NLS Dataset 2');
subplot(3, 2, 3); plotpv(P3, T3); title('NLS Dataset 3');
subplot(3, 2, 4); plotpv(P4, T4); title('NLS Dataset 4');
subplot(3, 2, 5); plotpv(P5, T5); title('NLS Dataset 5');