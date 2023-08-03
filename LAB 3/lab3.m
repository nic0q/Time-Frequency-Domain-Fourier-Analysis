clearvars;
close all;
warning off;

% Inicialización de imágenes
image = imread("ejemplos\rut_1.jpg"); % Lectura del rut a evaluar
image_gray = rgb2gray(image); % Transformación a escala de grises
threshold = otsu(image_gray);
image_otsu =image_gray > threshold; % Binarización Otsu
image_bin = imbinarize(image_gray, "adaptive"); % Binarización Adaptativa
celda = cc(image_otsu); % Celda con los componentes conexos, bounding boxes

figure,
subplot(2,3,1), imshow(image_gray), title('original'); % Original 
subplot(2,3,2), imshow(image_otsu), title(strcat('Otsu umbral: ',num2str(threshold)));
subplot(2,3,3), imshow(imbinarize(image_gray, "adaptive")), title('Binarización Adapatativa');
subplot(2,3,4), imshow(bb(image, celda)), title('Componentes Conexos');
subplot(2,3,5), imshow(edge(image_bin)), title('Bordes Componentes');

% Vectores de características
n_modelos = 11;
n_digitos = 9;
v_modelos = get_modelos(n_modelos);
v_digitos = get_digit_features(n_digitos, image_gray, celda);
v_sub = zeros(16, n_modelos);

% Genera los histogramas de 16 cuadrantes
generate_hist(v_digitos)
generate_hist(v_modelos)

% Identificación de rut
rut = get_rut(v_modelos, v_digitos);
disp(strcat("EL rut identificado es: ", rut))
write_file(rut)

function write_file(rut)
    fid = fopen( 'rut.txt', 'wt' );
    fprintf( fid, '%s', rut);
    fclose(fid);
end

% Obtiene el rut "string" dados los modelos y dígitos
function str = get_rut(v_modelos, v_digitos)
    str = "";
    for k = 1:9
        sumas = zeros(11);
        for l = 1:11
            for h = 1:16
                dif = abs(v_modelos(h, l) - v_digitos(h, k));
                sumas(l) = sumas(l) + dif;
            end
        end
        minimo = min(sumas);
        index = find(sumas == minimo(1));
        if index == 11
            digit = "K";
        else
            digit = string(index - 1);
        end
        str = strcat(str, digit);
    end
end

% Obtiene las características de cada dígito identificado
function v_digitos = get_digit_features(n_digitos, image_gray, celda)
    v_digitos = zeros(16, n_digitos);
    for i = 1:9
        img = get_digito(image_gray, i, celda);
        img = edge(imbinarize(imresize(img, [128 64]), "adaptive"));
        v_digitos = features(img, v_digitos, i);
    end
end

% Función que obtiene un arreglo con los datos representativos de cada
% modelo
function v_modelos = get_modelos(n_modelos)
    v_modelos = zeros(16, n_modelos);
    for i=1:n_modelos
        if i == 11 % Va a la carpeta dígito verificador "K"
            location = "modelos\K\*.jpg";
        else
            location = "modelos\" + (i - 1) + "\*.jpg";
        end
        ds = imageDatastore(location);
        for j = 1:numel(ds.Files)
            v_temp = zeros(16, 1);        % Arreglo temporal
            instancia = readimage(ds, j);
            instancia = padarray(imcomplement(instancia), [1,1], 0, 'both');
            c2 = imbinarize(imresize(instancia, [128 64]), "adaptive");
            comp = imcomplement(c2);
            instancia = edge(comp);
            v_temp = features(instancia, v_temp, 1);
            for k = 1:16 % Acumula de cada instancia
                v_modelos(k, i) = v_modelos(k, i) + v_temp(k, 1);
            end
        end
        for k = 1:16 % Obtiene el promedio para cada modelo
            v_modelos(k, i) = v_modelos(k, i) / numel(ds.Files);
        end
     end
end

% Genera el histograma de 16 cuadrantes
function generate_hist(vector)
    figure,
    digit = 0;
    for i = 1:size(vector, 2)
        subplot(4,3, i), histogram(vector(:,i), 16), title(digit);
        digit = digit + 1;
    end
end

% Corta una imagen dado un índice de dígitos del arreglo
function img = get_digito(image, index, celda)
    img = imcrop(image, celda{4, index});
end

% Obtiene las características de una imagen y devuelve un arreglo
function vc = features(img, vc, index)
    qua = 1;
    for i = 1:4 % 4 cuadrantes grandes
        Q = getQuadrant(img, i);
        % 4 cuadrantes chicos
        for l = 1:4
            Q_sub = getQuadrant(Q, l);
            ones = 0; % Numero de ceros por cuadrante
            size_q = 32 * 16; % Tamaño del cuadrante
            for j = 1:32 % Itera dentro del cuadrante
                for k = 1:16
                    if Q_sub(j, k) == 1
                        ones = ones + 1;
                    end
                end
            end
            vc(qua, index) = (ones / size_q) * 100;
            qua = qua + 1;
        end
     end
end

% Algoritmo del umbral de Otsu
function threshold = otsu(img)
    hist = histogram(img,'Normalization','probability','binwidth',1,'BinLimits',[0,255]);
    max = realmax; % Valor maximo como referencia
    % Formula usada
    for t = 1:255
        Q1 = sum(hist.Values(1:t)); %% Primer componente formula "q"
        Q2 = sum(hist.Values(t+1:255));
        M1 = sum ((1:t).*hist.Values(1:t))/Q1; % Segundo sumando "Mu"
        M2 = sum((t+1:255).*hist.Values(t+1:255))/Q2;
        S1 = sum(((((1:t)-M1).^2).*hist.Values(1:t)))/Q1; % Tercer, "sigma"
        S2 = sum(((((t+1:255)-M2).^2).*hist.Values(t+1:255)))/Q2;
        Sw = Q1*S1+Q2*S2; % "Sigma w"
        if max > Sw
            max = Sw; % Varianza mínima
            threshold = t; % Umbral
        end
    end
end

% Obtiene uno de 4 cuadrantes que está dividida la imagen
function Q = getQuadrant(I, q)
    if q == 1
        Q = I(1:size(I,1)/2,1:size(I,2)/2,:);
    end
    if q == 2
        Q = I(size(I,1)/2+1:size(I,1),1:size(I,2)/2,:);
    end    
    if q == 3
        Q = I(1:size(I,1)/2,size(I,2)/2+1:size(I,2),:);
    end
    if q == 4
        Q = I(size(I,1)/2+1:size(I,1),size(I,2)/2+1:size(I,2),:);
    end
end

% Componentes conexos
% id | puntos del componente | bordes | [iniciox, inicioy, ancho, largo]
function celda = cc(image_bin)
    % Se identifican los componentes conexos
    cc = bwconncomp(imcomplement(image_bin));
    celda = cell(4,9);
    position = 0;
    for i = 1:cc.NumObjects % Por cada componente conexo identificado
        % Obtener las coordenadas (posiciones) de los píxeles utilizando ind2sub
        [rows, cols] = ind2sub(cc.ImageSize, cc.PixelIdxList{i});
        points = [rows, cols];
        xmin = min(cols);
        ymin = min(rows);
        width = max(cols) - min(cols);
        height = max(rows) - min(rows);
        % Ignora guiones y puntos si el area de la Bounding Box es menor 40
        if height * width < 40
            continue
        end
        position = position + 1; % Digitos
        % Imagen cortada de cada dígito del rut
        simbolo = imcrop(image_bin,[xmin-1 ymin-1 width+2 height+2]);
        simbolo = imresize(simbolo, [128 64]);
        lista = {position, points, edge(simbolo), [xmin-1 ymin-1 width+2 height+2]};
        for j = 1:length(lista)
            celda(j, position) = lista(j);
        end
    end
end

% Colorea las bounding box y retorna la imagen
function image = bb(image, celda)
        for i = 1:length(celda)
            vector = celda{4, i};
            xmin = vector(1);
            ymin = vector(2);
            height = vector(3);
            width = vector(4);
            for j = xmin - 1:(xmin + height + 1)
                image(ymin - 1, j, :) = [255, 52, 25];
                image(ymin + width + 1, j, :) = [255, 52, 25];
            end
            for k = ymin - 1:(ymin + width + 1)
                image(k, xmin - 1, :) = [255, 52, 25];
                image(k, xmin + height + 1, :) = [255, 52, 25];
            end
        end
end
