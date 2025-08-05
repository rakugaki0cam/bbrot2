#   bbrot.py
#                   v2
#
#   2022.07.09
#
#   変更履歴
#   2022.10.10  OCR仕様変更ー表示器をdeBUGgerからOLEDに変更のため。
#   2022.10.28  回転方向を限定することで-359〜359度の回転解析角度とした。
#               解析の時の角度を前回値から推定することで解析角度を狭めて高速化。
#               解析結果をcsvファイルに書き出し。
#   2022.10.29  OLED表示フォントを作ったのでOCR読み取り精度が上がった。ほぼOK。->読み取り自動化。
#   2023.04.18  角度0度の時の異常値判定を回避
#   2023.05.08  bbpictディレクトリ内の画像ファイル読込の際、子ディレクトリを除外しエラー回避
#   2023.05.08  読み取りエラーの時、解析角度範囲を狭めてやり直し...少しは救われるデータあり。時間かかる以外悪くなるところはないので採用
#   2023.05.09  BB弾回転アニメ.gifファイルを出力 
#   2023.07.23  python3.11だとエラーになる。3.10.9でOK
#   2023.10.17  gifのdurationをdtに比例させ1/600スロー再生になるようにした。
#   2025.08.05  gitがわけわからん　データが消える  一度にアップデートできる量を超えているよう　設定を変更　git config --global http.postBuffer 157286400　してOK
#

#ライブラリの読み込み
import sys
import os
import math
#from webbrowser import BackgroundBrowser
import csv
import cv2

import pyocr
import pyocr.builders
#import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/path/to/dir')

#ウインドウの設定
cv2.namedWindow('template3', cv2.WINDOW_NORMAL)
cv2.moveWindow('template3', 200, 200)
cv2.namedWindow('match BB', cv2.WINDOW_NORMAL)
cv2.moveWindow('match BB', 0, 200)

#色の定義
mazenta = (255, 0, 255)
yellow  = (0, 255, 255)
green   = (0, 255, 0)
darkGreen = (0, 127, 0)
white   = (255, 255, 255)

def process(filename, mode):
    """
    写っているすべてのBB弾の角度を推定

    Parameters
    ----------
    filename : string
        画像ファイル名
    mode: string
        360standard - 標準角度範囲-360〜360度    
        180redo ----- 範囲を-180〜180度にせばめてやり直し
        90redo ----- 範囲を-90〜90度にせばめてやり直し
    Returns
    -------
        結果をオーバーレイした画像
    """
    global statusE, incAngle, txtDt, txtV0, dt, v0

    image = cv2.imread(filename)
    #画像サイズと推定BB弾寸法（撮影範囲による）
    #撮影対象の大きさ[mm]
    subjectWidthMin = 210   #横方向撮影範囲寸法
    subjectWidthMax = 320
    bbObjectSize = 6        #BB弾の物理寸法
    #BB弾の推定サイズ[pixel]の計算
    bbPixelMin = int(image.shape[1] / subjectWidthMax * bbObjectSize)
    bbPixelMax = int(image.shape[1] / subjectWidthMin * bbObjectSize)
    #print('画像サイズ (x:', image.shape[1], 'y:', image.shape[0], ')', end = ' ')
    #print('検出するBB弾の大きさ', bbPixelMin, '~', bbPixelMax, 'pix')

    #グレースケール化(OCRで使用)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #高さを中央付近450pxにクロップ
    w = image.shape[1]          #GX8:5184, GX7:4592
    h = int(image.shape[0] * 0.116) # GX8:3888, GX7:3448 -> 450pixel切り取り画像の縦ピクセル数 ### 2022/10/10 撮影高さ失敗600にて可
    top = (image.shape[0] - h) // 2
    flip = image[top: top + h, 0: w]
    #右から撃っているので、左から右への時系列になるように左右反転
    flip = cv2.flip(flip, 1)                                        # =0:上下反転、>0:左右反転、 <0:上下左右反転
    #コントラスト調整 (テンプレートマッチングで使用)
    scaled = cv2.convertScaleAbs(flip, alpha = 3.5, beta = 0)       #alpha:スケールファクタ1.0〜2.0 beta:加算値
    #ノイズ除去してブロブ検出用の画像を作成
    median = cv2.medianBlur(flip, ksize = 9)                        #ksizeは奇数 円の検出大きさに関わってくる
    #平坦化 （オプション）
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    cl1 = clahe.apply(flip)
    #二値化の閾値を求める
    minGray = 0
    maxGray = 255
    #大津の手法　minGrayに大津の閾値が入る
    minGray, threshold = cv2.threshold(median, minGray, maxGray, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #各処理画像の表示
    hw = 350    #ディスプレイ上の表示座標y
    hwd = 140
    winShow(flip, 'flip', (700, hw), (w // 4, h // 4))  #1/4に縮小
    winShow(scaled, 'scaled', (700, hw + hwd), (w // 4, h // 4))
    winShow(median, 'median', (700, hw + hwd + hwd), (w // 4, h // 4))
    winShow(threshold, 'threshold', (700, hw + hwd + hwd + hwd), (w // 4, h // 4))
    winShow(cl1, 'CLAHE', (700, hw + hwd + hwd + hwd + hwd), (w // 4, h // 4))
    #print('ブロブ検出用二値化　大津の閾値 = ',minGray)

    #ヒストグラム
    histG = False  #True
    if histG is True:
        hist1 = cv2.calcHist([flip], channels = [0], mask = None, histSize = [256], ranges = [0,256])
        hist2 = cv2.calcHist([scaled], channels = [0], mask = None, histSize = [256], ranges = [0,256])
        hist3 = cv2.calcHist([cl1], channels = [0], mask = None, histSize = [256], ranges = [0,256])
        plt.plot(hist1)  #ヒストグラム
        plt.plot(hist2)  #ヒストグラム
        plt.plot(hist3)  #ヒストグラム
        plt.show()


    ######################### 検出方法を選択 #####################################################
    flipBGR = cv2.cvtColor(flip.copy(), cv2.COLOR_GRAY2BGR)   #イメージ出力の元画像

    #detectMethod = "Blobs" #少し小さくなるようだ。検出できないことも多い。
    detectMethod = "Hough"
    #if detectMethod == 'Blobs':
    #    #Blob円検出による
    #    bbImg, bbData = circlesBlobs(flipBGR, threshold, minGray, maxGray, bbPixelMin, bbPixelMax) #　ブロブ円検出
    #else:
        #Hough円検出による
    bbImg, bbData = circlesHough(flipBGR, median, bbPixelMin, bbPixelMax) #　ハフ円検出

    if bbData is None:
        statusE = '円検出できず'
        print(statusE)
        winShow(bbImg, detectMethod, (700, 190), (w // 4, h // 4))  #1/4に縮小
        cv2.waitKey(1)
        return bbImg, 0, 0, 0, 0, statusE


    winShow(bbImg, detectMethod, (700, 190), (w // 4, h // 4))  #1/4に縮小
    cv2.waitKey(1)

    bbCount = len(bbData)
    print('BB検出数', bbCount, end = '  ')

    #適正サイズ外のデータを除外  ########### 除外すると計算が変
    #外周にマークが被りすぎると円検出が小さくなるようだ
    bbData2 = excludeSmall(bbData)                  #現在は未使用#######
    bbCount2 = len(bbData2)

    ### 途中が抜けた時の処理 ーーー 未定 #######################################################

    print('適正BB数', bbCount2)
    #BB検出数の判定
    if bbCount < 10:
        statusE = 'BB検出数が不足'
        input(statusE)
        return bbImg, 0, 0, 0, 0, statusE

    #print('BBデータ [n,x,y,r]: ', bbData)

    #bb進行方向の傾斜角度の計算
    dX, dY, incAngle = inclinationAngle(bbData)
    print(f'傾斜角度: {dY:6.3f}/{dX:6.1f} = {incAngle:6.3f}deg')


    #BB弾検出結果から作業用画像を再度正確にクロップする
    bbImg = cv2.cvtColor(flip.copy(), cv2.COLOR_GRAY2BGR)   #イメージ出力の元画像を再作成
    bbImg, top, left = bbImageCrop(bbImg, bbData)
    scaled, _ , _    = bbImageCrop(scaled, bbData)  #マッチ検出用画像
    median, _ , _    = bbImageCrop(median, bbData)  #median画像

    #クロップ後の座標修正
    for i in range(bbCount):
        bbData[i][1] -= left
        bbData[i][2] -= top

    #print('BBデータ [n,x,y,r]: ', bbData)

    #BB弾の回転アニメをGIFファイルで出力
    cv2Image = []
    gifImages = []
    for b in bbData:
        imgCrop = crop(bbImg, (b[1],b[2]), (150,150))  #BB玉をクロップ
        cv2Image.append(imgCrop)                #cv2 imageのまま保存

        imgPillow = Image.fromarray(imgCrop)    #pillow用画像に変換
        gifImages.append(imgPillow)             #gif image
    # セーブは発光間隔dtをOCRで取得後
    
   #BB回転アニメーション...GIF
    for im in cv2Image:
        winShow(im, 'gif', (240, 200), (100, 100))
        cv2.waitKey(100)

    #円周と中心点を描画
    for b in bbData:
        cv2.circle(bbImg, (b[1], b[2]), b[3], green, thickness = 1)
        cv2.drawMarker(bbImg, (b[1], b[2]), darkGreen, markerType = cv2.MARKER_CROSS, markerSize = 300, thickness = 1)
    cv2.waitKey(1)



    #### OCR ########################################################
    txtDt, txtV0 = ocrLcd(image)

    #Δ時間の読み取り値が数値変換時にエラーの時（　.がコンマだったり、数字じゃなかったり）
    try:
        dt = float(txtDt)
    except ValueError:
        dt = 1000000    #ゼロだとエラーでとまるため大きい数値に
    print(f'周期時間読取テキスト "{txtDt}" ', end = ' ')
    try:
        txtV0 = float(txtV0)
    except ValueError:
        print(f'*** 読み取り初速値がエラー "{txtV0}"   周期時間値がOKの時はreturnキーを押す')
        txtV0 = 99999

    #初速の計算で周期値の読み取りが正しいかを確認
    v0 = 0.012 / (dt / 1000000)
    if abs(txtV0 - v0) >= 0.1:
        #初速の計算値が合わない時
        if txtV0 != 99999:
            print(f'*** 読み取り初速値 "{txtV0}"と計算値{v0:6.2f}m/sが合わない  周期時間値がOKの時はreturnキーを押す')
        #周期をキー入力
        while True:
            inpTxt = input('コマ間の周期時間 [usec] = ')
            dtMin = 100     #usec  @v0=120m/sec
            dtMax = 1000    #usec  @v0=12m/sec
            if inpTxt == '':
                dt = float(txtDt)
                if (dt > dtMin and dt < dtMax):     #12 ~ 100m/sec (@12mm)
                    break
                else:
                    continue
            dt = float(inpTxt)
            if (dt > dtMin and dt < dtMax):
                v0 = 0.012 / (dt / 1000000)         #初速の計算を入力値でやり直し
                break
    #OK
    print(f'  周期dt = {dt:6.2f}usec', end = " ")   #周期値
    print(f'  初速v0 = {v0:6.2f}m/sec')             #初速

    #画像へコマ周期、初速、ファイル名を書き込み
    text = f"{detectMethod} circles:{bbCount:2d}  v0:{v0:6.2f}m/sec  dt:{dt:6.2f}usec  incline:{incAngle:6.3f}deg  ({filename})"
    locate = (int(bbData[0][1]), int(bbData[0][2] + 100))
    cv2.putText(bbImg, text, locate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 1)

    cv2.imshow("BB image", bbImg)
    cv2.imshow('scaled', scaled)
    cv2.imshow('median', median)
    cv2.waitKey(1)

    # GIF セーブ
    # pillow画像からGIFファイルにしてセーブ　（ストロボ発光間隔dtをOCRで取得後にセーブ）
    slowRate = 600                      #1/600スロー再生
    gifDtMsec = dt * slowRate / 1000    #gifの周期時間msec　dtはusec
    imageNum = os.path.splitext(os.path.basename(filename))[0]    #拡張子無しファイルネームを切り出し
    gifImages[0].save(resultdir+'gif'+imageNum+'.gif', save_all=True, append_images=gifImages[1:], optimize=False, duration=gifDtMsec, loop=0)    



################　回転マッチング　############################

    #カメラ撮影光軸中心付近のBB弾をテンプレートとする
    indexCenter = bbCount // 2    #//の答えは整数となる 標準条件で7となる
    #標準条件: ストロボ発光間隔　初速12mm距離の時間、発光回数　15回

    tpCenter = (bbData[indexCenter][1], bbData[indexCenter][2])     #BB中心
    bbDia = bbData[indexCenter][3] * 2                              #BB直径

    kShadow = 0.94                  #0.93 # 2020/08/16 P110416~
    tpDia = int(bbDia * kShadow)    #周辺の影の部分をマスク
    imgc = crop(scaled, tpCenter, (tpDia, tpDia))
    template = mask_circle(imgc, tpDia)
    #2値化オプション??  ###
    #ret, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #########特徴点検出用テンプレート（周辺画像までいれて広くしないと特徴点が検出されない）
    #template2 = crop(median, tpCenter,(tpDia * 3, tpDia * 3))
    #####################特徴点検出法


    #検出したBB弾すべてに角度推定を実行する
    print()
    print('計算開始')

    result = []
    xt = []
    yt = []
    #回転方向を限定して、読取回転角を制限＝回転角がいったりきたりしなくなる->異常値エラーになる
    if mode == '360standard':
        startAngle = -350  #-358  
    elif mode == '180redo':
        startAngle = -178           #スタート角  2023.4.17 ほとんどかわらない（エラーを救えなかった）
    elif mode == '90redo':
        startAngle = -88            #スタート角    
    

    matchAngle = startAngle
    endAngle = 0
    firstAngle = 999

    for i in range(bbCount):
        point = (bbData[i][1], bbData[i][2])        #BB中心

        #回転角度を制限
        startAngle = int(matchAngle) - 1            #スタート角
        if i < indexCenter:
            endAngle = 0                            #エンド角
        elif i == indexCenter:
            # 右端の回転角を先に読み取り
            point2 = (bbData[bbCount-1][1], bbData[bbCount-1][2])   #右端のBB中心
            endAngle = int(- firstAngle * 1.2)                      #左端の読取角＋α
            lastAngle, _ = estimateRot(scaled, template, 0, endAngle, point2, bbDia)
            endAngle = 1
        else:
            endAngle = int(lastAngle) + 1           #右端BBの読取角
        #print("start {:4d}deg - end {:4d}deg".format(startAngle, endAngle))     #解析角度範囲

        #回転角を読み取り
        matchAngle, _ = estimateRot(scaled, template, startAngle, endAngle, point, bbDia)    # scaled
        #print("{:5.1f}deg  Match value:{:10.0f}".format(matchAngle, matchVal))

        #########################################################
        #特徴点検出法のとき
        #angle2 = contourMatch(median, template2, point, bbDia)
        #########################################################

        
        if firstAngle == 999:
            firstAngle = matchAngle     #最初のコマの角度
            prevAngle = matchAngle      #初回の代入で必要



        #変化角度の確認
        dAngle = matchAngle - prevAngle
        ###################################################
        

        #回転角度の線を表示
        lineLen = 160 // 2
        matchRad = math.radians(matchAngle)
        dl = (math.cos(matchRad) * lineLen, - math.sin(matchRad) * lineLen)
        pt1 = np.add     (point, dl).astype(np.int16)
        pt2 = np.subtract(point, dl).astype(np.int16)
        cv2.line(bbImg, tuple(pt1), tuple(pt2), color = mazenta, thickness = 1, lineType = cv2.LINE_AA, shift = 0)  

        #角度の書き込み
        wholeAngle = matchAngle - firstAngle
        prevAngle = matchAngle

        txtMatchAng = f'{matchAngle:6.1f}deg'
        cv2.putText(bbImg, txtMatchAng, org = (point[0] + 30, point[1] - 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = mazenta, thickness = 1)
        if i >= 1:
            textdAng = f'({dAngle:5.1f})'
            cv2.putText(bbImg, textdAng, org = (point[0] + 65, point[1] - 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = yellow, thickness = 1)

        #回転角の書き込み
        try:
            rotPs = 1000000.0 / dt / (i * 360 / wholeAngle)
        except ZeroDivisionError:
            rotPs = 0

        if i > indexCenter:
            textRps =  f'{rotPs:6.1f}rps'
            cv2.putText(bbImg, textRps, org = (point[0] + 0, point[1] + 90), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = white, thickness = 1)
            #中心振り分けでの回転数計算
            dk = i - indexCenter
            try:
                rotPs2 = 1000000.0 / dt /((dk * 2) * 360 /(matchAngle - result[indexCenter - dk][5]))
            except ZeroDivisionError:
                rotPs2 = 0

            textRps =  f'{rotPs2:6.1f}rps'
            cv2.putText(bbImg, textRps, org = (point[0] + 10, point[1] + 108), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = yellow, thickness = 1)

        result.append((i, i * dt, point[0], point[1], bbData[i][3], matchAngle, dAngle, wholeAngle, rotPs))
        xt.append(i * dt)
        yt.append(matchAngle)
        r = result[-1]
        if len(result) > 1:
            dx = r[2] - result[-2][2]
        else:
            dx = 0

        print(f'{(r[0] + 1):2.0f}   t:{r[1]:8.2f} x:{r[2]:5.0f} dx:{dx:5.0f} y:{r[3]:5.0f} r:{r[4]:3.0f}   {r[5]:6.1f}deg ({r[6]:6.1f})  total {r[7]:6.1f}deg   {rotPs:6.1f}rps')

        cv2.imshow("BB image", bbImg)
        cv2.waitKey(1)


    # 回帰計算により回転数を求める
    px = result[indexCenter][2] - 60      #文字の表示位置
    py = result[indexCenter][3] + 100

####### 回帰直線から2デルタ以上離れているものを除外して、もう一度回帰計算する #######################　未


    # 異常値の検出　da
    aveDa = result[-1][7] / (bbCount - 1)
    print(f'コマ間角度変化分の平均値 {aveDa:6.1f}°')

    numDel = 0
    for i, r in enumerate(result):
        if i == 0:
            continue
        limUp = aveDa * 1.2     #aveDaが小さいときにだめなので1度にする
        if limUp < 1:
            limUp = 1
        limLow = aveDa * 0.8    #aveDaが小さいときマイナスになることもあるので-1度として読み取り角度が0.0の時にエラーにさせない
        if limLow < 1:
            limLow = -1
        if r[6] > limUp or r[6] < limLow:
            print(f'{(i + 1):2d}コマ目 - {r[6]:6.1f}° は異常値のため除外')      #aveDaがマイナスの時（回転が微妙かゼロの時）に異常値判定になってしまう。
            del xt[i - numDel]
            del yt[i - numDel]
            numDel += 1     #デリートすると1つ詰まって、添字の位置が変わる
            if numDel > 8:  #9つ目で中止
                print("***** 読み取り不可 *****")
                cv2.putText(bbImg, 'XXXXXX', org = (px, py), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = white, thickness = 2)
                cv2.imshow("BB image", bbImg)
                cv2.waitKey(3000)

                #範囲を狭めて再計算させる
                if (mode == '360standard'):
                    #画像解析角度範囲をー180〜180度に制限してもう一度やり直しする　################
                    print("***** 解析範囲を-180〜180度にせばめて再計算します")
                    statusE = "180redo"
                    return bbImg, incAngle, v0, dt, "---", statusE 
                elif (mode == '180redo'):
                    #画像解析角度範囲をー90〜90度に制限してもう一度やり直しする　################
                    print("***** 解析範囲を-90〜90度にせばめて再計算します")
                    statusE = "90redo"
                    return bbImg, incAngle, v0, dt, "---", statusE
                return bbImg, incAngle, v0, dt, "---", "error"
            
    if numDel > 0:
        print(f'除外データ数: {numDel:1d}個')

    # 回帰計算
    a,b = np.polyfit(xt, yt, 1)     # y=ax+b

    print(f'直線回帰式 y = {a:7.4f}x {b:+8.2f}   y:角度  x:時刻 ')
    x0 = result[0][0]
    x1 = result[bbCount - 1][0]
    y0 = a * x0 + b
    y1 = a * x1 + b
    rotReg = (y1 - y0) / 360 / ((x1 - x0)/ 1000000)
    print(f'ホップ回転数{rotReg:6.1f}rps (回帰計算による)')
    textRps = f'{rotReg:6.1f}rps'

    cv2.putText(bbImg, textRps, org = (px, py), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = white, thickness = 2)
    cv2.imshow("BB image", bbImg)

    ##ステータス
    if numDel == 0:
        statusE = ''
    else:
        statusE = f'Err{numDel:1d}'

    print ('計算完了')

    #################
    #cv2.waitKey(200) ##################### 0: WAIT ############### 200: moment
    ##################



    return bbImg, incAngle, v0, dt, rotReg, statusE



###### BB円検出　＆　データ整形 ######################################

def circlesHough(image, median, bbPixelMin, bbPixelMax):
    """
    ハフ円でBB弾を検出

    Parameters
    ----------
    image : mat
        出力画像の元
    median : mat
        検出する画像
    
    Returns
    -------
        円検出画像

        円位置座標とサイズ
    """
    #　ハフ円検出
    #円検出
    circles = cv2.HoughCircles(median, cv2.HOUGH_GRADIENT, dp = 1.0, minDist = 200, param1 = 100, param2 = 40, minRadius = bbPixelMin // 2-10, maxRadius = bbPixelMax // 2+20)
    if circles is None:
        #検出ゼロのとき
        bbData = None
        return image, bbData

    #型変換
    circles = np.round(circles[0, :]).astype('int16')

    #円を描写
    for (x, y, r) in circles:
        # 円周を描画する
        cv2.circle(image, (x, y), r, green, thickness = 1)
        # 中心点を描画する
        cv2.drawMarker(image, (x, y), darkGreen, markerType = cv2.MARKER_CROSS, markerSize = 300, thickness = 1)

    #x位置順にソート
    sortedCircles = sorted(circles, key=lambda x: x[0])   #x[0]:x座標 で並べ替えて新たな変数へ代入
    #BBデータを整理
    bbData = []        #(n, x, y, r)
    bbnum = 0
    for c in sortedCircles:
        bbData.append([bbnum, c[0], c[1], c[2]])
        bbnum += 1

    #円の個数
    #bbcount = len(bbData)
    #print(f'Hough circles: {bbcount}')
    #print(bbData)
    return image, bbData


'''
def circlesBlobs(image, threshold, minGray, maxGray, bbPixelMin, bbPixelMax):
    """
    ブロブ円でBB弾を検出

    Parameters
    ----------
    image : mat
        出力画像の元
    threshold : mat
        検出する画像
    minGray : int
    maxGray : int
        二値化の閾値 0 ~ 255
    bbPixelMin : int
    bbPixelMax : int
        BB弾の大きさ px
    
    Returns
    -------
        円検出画像
        円位置座標とサイズ
    """
    #ブロブ円検出
    keypoints = blobsDetect(threshold, minGray, maxGray, bbPixelMin, bbPixelMax)
    if len(keypoints) == 0:
        #検出ゼロのとき
        bbData = None
        return image, bbData

    #ブロブを円形で表示
    image = cv2.drawKeypoints(image, keypoints, image, green, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #x位置順にソート
    keypoints = sorted(keypoints, key=lambda kp: kp.pt[0])    #kp.pt[0]:x座標
    #BBデータを整理
    bbData = []        #(n, x, y, r)
    bbnum = 0
    for k in keypoints:
        point = (int(k.pt[0]), int(k.pt[1]))
        #ブロブの中心を表示
        cv2.drawMarker(image, point, darkGreen, markerType = cv2.MARKER_CROSS, markerSize = 300, thickness = 1)
        bbData.append([bbnum, point[0], point[1], int(k.size / 2)])
        bbnum += 1

    #ブロブの個数
    #bbcount = len(bbData)
    #print(f'blob circles: {bbcount}')
    #print(bbData)
    return  image, bbData


def blobsDetect(image, minGray, maxGray, bbPixelMin, bbPixelMax):
    """
    ブロブ（かたまり）検出器を設定して検出

    Parameters
    ----------
    image : mat
        検出対象の画像
    minGray : int
    maxGray : int
        二値化の閾値 0 ~ 255
    bbPixelMin : int
    bbPixelMax : int
        BB弾の大きさ px
    
    Returns
    -------
        ブロブ位置座標とサイズ
    """

    #ブロブ（塊）検出器の設定
    params = cv2.SimpleBlobDetector_Params()
    #白のブロブを検出
    params.blobColor = 255      #黒検出したい時は0
     #閾値
    params.minThreshold = minGray
    params.maxThreshold = maxGray
    #塊の大きさ（面積）
    params.filterByArea = True
    shade = 0.95      #影、インク模様の面積分の補正
    params.minArea = bbPixelMin ** 2 * math.pi / 4 * shade      ##小さい円が入りすぎ＃＃＃＃＃＃＃＃＃＃＃
    params.maxArea = bbPixelMax ** 2 * math.pi / 4
    #円形度でフィルタ(凹面concave)
    params.filterByCircularity = True
    params.minCircularity = 0.4   #0〜1 = 4πS/L^2   S:面積(画素数) L:周囲長　　　円形度が高い->1.0
    ###模様が外周にかかって輪郭が切れた時に凹面になるので小さめの値にする
    #凸面フィルタ
    params.filterByConvexity = True
    params.minConvexity = 0.5       #0〜1 = S/C  S:面積　C:凸面の面積（円形から出っぱった分）
    #楕円形フィルタ（形態の伸び　円形=1、直線=0 慣性モーメント）
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    #検出器を設定
    ver = (cv2.__version__).split('.')
    if int(ver[0]) <= 2:
        #openCV ver.2
        detector = cv2.SimpleBlobDetector(params)
    else:
        #openCV ver.3~
        detector = cv2.SimpleBlobDetector_create(params)

    #検出器を作動（ブロブを検出する）
    keypoints = detector.detect(image)
    return keypoints
'''


###### データ整形関数　#################

def excludeSmall(bbData):
    """
    BBサイズが平均から離れているものを除外する

    Parameters
    ----------
    bbData
        bbデータ [n,x,y,r]

    Returns
    -------
        選別後のbbデータ [n,x,y,r]
    """

    #BB円平均サイズ(半径で計算、表示は直径)を求める
    bbCount = len(bbData)
    sumSize = 0
    for b in bbData:
        sumSize += b[3]
    bbAve = sumSize / bbCount
    print(f'BB平均直径 = {(bbAve * 2):4.1f}px', end = ' ')

    #判定サイズの計算
    bbMin = 0.95 * bbAve
    bbMax = 1.05 * bbAve
    #print('適正範囲 = {:4.1f} ~ {:4.1f}px'.format(bbMin * 2, bbMax * 2), end = '')
    print()
    #選別
    bbData2 = []
    newIndex = 0
    for i, b in enumerate(bbData):
        #小さすぎるものを除外する
        if (b[3] > bbMax) or (b[3] < bbMin):
            print(f'No.{i:2.0f}  {(b[3] * 2):5.1f}px はサイズ不適合')
        else:
            #合格のデータだけで再構成
            bbData2.append([newIndex, b[1], b[2], b[3]])
            newIndex += 1

    return bbData2


def  inclinationAngle(bbData):
    """
    BBの進行方向の傾き
    上向が＋

    Parameters
    ----------
    bbData : [n,x,y,r]
        bbデータ

    Returns
    -------
        x距離
        y変位
        進行方向の角度（仰俯角）
    """

    global incAngle

    num = len(bbData)
    dY = float(bbData[num - 1][2]) - float(bbData[0][2])
    dX = float(bbData[num - 1][1]) - float(bbData[0][1])
    incAngle = -math.degrees(math.atan(dY / dX))
    return dX, dY, incAngle


def bbImageCrop(image, bbData):
    """
    BBサイズから作業用画像を正確にクロップ

    Parameters
    ----------
    image : mat
    bbData : [n,x,y,r]
        bbデータ

    Returns
    -------
        クロップした画像
        topオフセット値
        leftオフセット値
    """

    bbCount = len(bbData)

    #x座標の計算
    offsetX = 250

    left = bbData[0][1] - offsetX
    if left < 0:
        left = 0
    right = bbData[bbCount - 1][1] + offsetX
    if right > image.shape[1]:              #(y_size, x_size, color)
        right = image.shape[1]

    #y座標の計算
    offsetY = 85
    bby = sorted(bbData, key = lambda x: x[1])    #y座標でソート

    top = bby[0][2] - bbData[0][3] - offsetY
    if top < 0:
        top = 0
    bottom = bby[bbCount - 1][2]  + bbData[0][3] + offsetY  #(y_size, x_size, color)
    if bottom > image.shape[0]:
        bottom = image.shape[0]

    #画像をクロップ
    d = len(image.shape)
    if d <= 2:  #gray or color
        image = image[top:bottom, left:right]
    else:
        image = image[top:bottom, left:right, :]

    return image, top, left





##### テンプレート or 特徴点 マッチング ############################################

def estimateRot(image, template, startAngle, endAngle, pt, size):
    """
    回転角の推定

    Parameters
    ----------
    image : mat
        推定対象の画像
    template : mat
        テンプレート画像
    startAngle :
        解析開始角度
    endAngle :
        解析終了角度
    pt : [number, number]
        推定対象の中心位置[col,row]
    size : any
        比較するBB画像の直径 px

    Returns
    -------
    マッチング角度 -360~+360(deg)
    マッチング値
    """
    angle = -1
    max9 = -9999999999
    ext = 10    #周りを少し広く
    cr = crop(image, pt, (size + ext, size + ext ))
    #周囲を白く
    k = 0.94    #縁が出ないように ##################2022.08.19
    cr = mask_circle(cr, int(size * k))

    cv2.imshow("match BB", cr)
    (w, h) = template.shape[: : -1]

    #### whole test #####
    # img2 = image.copy()
    # matchResult = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF)
    # _, maxVal, _, maxLoc = cv2.minMaxLoc(matchResult)   ###minVal,maxVal,minIndex,maxIndex
    # topLeft = maxLoc
    # bottomRight = (topLeft[0] + w, topLeft[1] + h)
    # cv2.rectangle(img2, topLeft, bottomRight, (255, 0, 255), 3)
    # マッチリザルト
    # plt.subplot(121),plt.imshow(matchResult, cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img2, cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # うまくいかない
    #
    #######################


    # ぐるぐる回しながら相関が最大となる角度を求める
    denomi = 10      #角度計算のステップ　1度/denomi  2->0.5deg 10->0.1deg
    matchAngle = 0
    for i in range(startAngle * denomi, endAngle * denomi):
        angle = i / denomi
        
        tp = rot(template, angle)
        matchResult = cv2.matchTemplate(cr, tp, cv2.TM_CCOEFF)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(matchResult)   ###minVal,maxVal,minIndex,maxIndex

        if maxVal > max9:
            cr2 = cr.copy()
            cr2 = cv2.cvtColor(cr2, cv2.COLOR_GRAY2BGR)
            matchAngle = angle
            max9 = maxVal
            if denomi <= 2:
                #細かく計算する時は表示しない
                #角度表示
                cv2.putText(tp, str(matchAngle), org = (0, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (0, 0, 0), thickness = 1)
                # org は左下の座標
                #マッチ値表示
                cv2.putText(tp, str(int(max9)), org = (0, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (0, 0, 0), thickness = 1)
                topLeft = maxLoc
                bottomRight = (topLeft[0] + w, topLeft[1] + h)
                cv2.rectangle(cr2, topLeft, bottomRight, (255, 0, 255), 1)

                cv2.imshow("match BB", cr2)
                cv2.imshow("template3", tp)
                cv2.waitKey(1)

                #マッチ　類似度のヒートマップ
                #fig, ax = plt.subplots(figsize=(10, 5))
                #im = ax.imshow(matchResult, cmap="jet")
                #plt.show()

    return matchAngle, max9


'''
def contourMatch(image, template, pt, size):
    """
    特徴点検出にて角度を求める

    Parameters
    ----------
    image : mat
        BB弾列の画像
    template : mat
        テンプレート画像
    pt : [number, number]
        推定対象の中心位置[col,row]
    size : any
        比較するBB画像の直径 px

    Returns
    -------
    マッチング角度 -180~+180(deg)#####################
    """

    ext = 200    #周りを少し広く
    cr = crop(image, pt, (size + ext, size + ext ))
    #ret, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #ret, cr = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #cv2.imshow("match BB2", template)
    match(template, cr)
    return


def match(img1, img2):
    # 特徴点抽出
    det = cv2.ORB_create()#ORB
    # 各画像の特徴点を取る
    kp1, des1 = det.detectAndCompute(img1, None)
    kp2, des2 = det.detectAndCompute(img2, None)

    # 2つの特徴点をマッチさせる
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck = True)
    #matches = bf.match(des1, des2)
    matches = bf.knnMatch(des1, des2, k = 2)

    # レシオテストを行う
    th = 0.85 #0.60
    good = []
    for m, n in matches:
        if m.distance < n.distance * th:
            good.append([m])

    # 特徴点を同士をつなぐ
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img1.copy())
    #cv2.imshow('keypoint match', img3)
    #cv2.waitKey(0)

    #変数の形を合わせる　＝　配列の次元と数を揃える
    nu = len(good)
    fromKp = []
    for k in kp1[:nu]:
        fromKp.append([k.pt[0],k.pt[1]])
    fromKp = np.reshape(fromKp, (-1, 1, 2))

    toKp = []
    for k in kp2[:nu]:
        toKp.append([k.pt[0],k.pt[1]])
    toKp   = np.reshape(toKp, (-1, 1, 2))

    aA, _ = cv2.estimateAffinePartial2D(fromKp, toKp)     #2つの変数の数が合わないとダメなよう

    #平行移動量
    # mM = aA[:2, :2]
    # t = aA[:, 2]
    # print(' M', mM, 't', t)
    # #回転角度
    degree = np.rad2deg(-np.arctan2(aA[0, 1], aA[0, 0]))
    print('特徴点 回転角度＝' , degree)

    textDa = 'f{degree:6.2f}deg'
    cv2.putText(img3, textDa, org = (160, 250), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = white, thickness = 1)

    cv2.imshow('keypoint match', img3)
    cv2.waitKey(1)


    return img3, (kp1, des1, kp2, des2, matches)
'''



##### OCR ##########################################################################

def ocrLcd(image):
    """
    OCRで発光周期時間を読み取り

    Parameters
    ----------
    image : mat
        画像

    Returns
    -------
        読み取った周期時間
        読み取った初速
    """

    #撮影時の表示の位置で設定を変更
    #if fileName < "P110169":

    ocrType = 4     # 0 --  2022/07/17~ 手入力  P1110155.JPG ~ P1110xxx.JPG
                    # 1 --  2022/07/17~ 手入力  P1110170.JPG ~ P1110236.JPG 　　確認　0170-179,0183-0191
                    # 1.2 - 2022/07/24~ 手入力  P1110241.JPG ~ P1110xxx.JPG 　　確認　0241-0256,0258-273
                    # 2 --  2022/08/16~  P1110414.JPG ~ P1110603.JPG
                    # 3 --  2022/10/10~  P1110609.JPG ~
                    # 4 --  2022/10/30~  P1110885.JPG ~

    if ocrType == 0:
        lcdImage = crop(image, ( 1800, 800) , ( 1000, 1000))
        lcdImage = cv2.convertScaleAbs(lcdImage, alpha = 0.5, beta = -20)
        lcdImage = cv2.medianBlur(lcdImage, ksize = 1) #ksizeは奇数
        ret, lcdImage = cv2.threshold(lcdImage, 95, 255,cv2.THRESH_BINARY)
    else:
        if ocrType == 1:
            # 2022/07/17~  P1110170.JPG ~ P1110xxx.JPG
            lcdImage = crop(image, ( 2250, 3250) , ( 850, 700))  #中心座標、サイズ
        if ocrType == 1.2:
            # 2020/07/24~  P1110241.JPG~
            lcdImage = crop(image, ( 1950, 3250) , ( 850, 800))  #中心座標、サイズ
        if ocrType == 2 or ocrType == 3:
            # 2022/08/16 P1110416.JPG~
            lcdImage = crop(image, ( 2400, 3250) , ( 700, 750))  #中心座標、サイズ
        if ocrType == 4:
            # 2022/10/30 P1110416.JPG~
            lcdImage = crop(image, ( 2350, 3200) , ( 700, 450))  #中心座標、サイズ

        #cv2.imshow('t',lcdImage)
        #cv2.waitKey(0)
        lcdImage = cv2.convertScaleAbs(lcdImage, alpha = 0.5, beta = -20)
        lcdImage = cv2.medianBlur(lcdImage, ksize = 3) #ksizeは奇数
        ret, _ = cv2.threshold(lcdImage, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #print('OCR用二値化 大津の閾値:', ret)
        ret, lcdImage = cv2.threshold(lcdImage, ret + 35, 255,cv2.THRESH_BINARY)
        #retはOTSUのときのしきい値
        kernel = np.ones((3,3),np.uint8)
        lcdImage = cv2.erode(lcdImage, kernel)

    #表示LCD(deBUGger)の回転
    if ocrType == 0:
        lcdImage = rot(lcdImage, 91.5)
    if ocrType >= 1 and ocrType <= 2:
        lcdImage = rot(lcdImage, -90)
    #2020/10/10~ 専用OLED LCDに変更したので回転不要


    #cv2.imshow("image4", lcdImage)
    #cv2.waitKey(1)
    lcdImage = cv2.bitwise_not(lcdImage)    #反転
    #cv2.imshow("image4", lcdImage)
    #cv2.waitKey(1)

    #####test 文字
    #cv2.putText(lcdImage, "* test 12.3m/sec (456.7us) test *", org = (20, 860), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = (0,0,0), thickness = 2)
    cv2.imwrite('ocrimg.png', lcdImage)

    cv2.namedWindow("lcdImage", cv2.WINDOW_NORMAL)
    h = lcdImage.shape[0]
    w = lcdImage.shape[1]
    rLcdImage = cv2.resize(lcdImage, (w // 2, h // 2))

    cv2.imshow("lcdImage", rLcdImage)
    cv2.moveWindow("lcdImage", 0, 500)

    #cv2.imwrite("test.png",lcdImage)
    cv2.waitKey(1)

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        #print("No OCR tool found")
        sys.exit()

    tool = tools[0]
    #print("will use tool '%s'" % (tool.get_name()))
    #langs = tool.get_available_languages()
    #print("available languages: %s" % ", ".join(langs))

    #builder = pyocr.builders.TextBuilder(tesseract_layout =  6) #text
    builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6) #box
    txt = tool.image_to_string(Image.open('ocrimg.png'), lang = 'eng', builder = builder)

    out = cv2.imread('ocrimg.png')
    for t in txt:
        #print('[ "' + t.content + '"', end = ' ')       # debug
        #print(t.position, "]")                          # debug
        cv2.rectangle(out, t.position[0], t.position[1], (255, 0, 255), 2)

    cv2.imshow('lcdImage', out)
    cv2.waitKey(1)
    #判別
    if len(txt) < 4:
        #読み取り項目不足のとき
        readVal = "xxx.x"
        readV0 = "xx.x"
        return readVal, readV0

    rvI = 0
    readVal = ""
    for i in range(-1, -3, -1):
        if txt[i].content == 'us':
            rvI = i - 1
            readVal = txt[rvI].content
            #print("dt読み取り値 ", readVal)    # debug
            break

    #usが識別できなかった時
    if rvI < 2:
        readVal = txt[-2].content
        readV0 = txt[-4].content
    else:
        readV0 = txt[rvI - 2].content
    #print("v0読み取り値 ", readV0)     # debug

    return readVal, readV0


##### サブ関数　##################################


# 画像の表示
def display(img, output_file_path):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()



def crop(image, pt, size):
    """
    画像の一部を矩形で切り取る

    Parameters
    ----------
    image : mat
        切り出し元の画像
    pt : [number, number]
        切り取り中心[col, row]
    size : [number, number]
        切り取りサイズ[width, height]

    Returns
    -------
        切り出した画像
    """
    left = int(pt[0] - size[0] / 2)
    if left < 0:
        left = 0
    right = int(pt[0] + size[0] / 2)
    top = int(pt[1] - size[1] / 2)
    if top < 0:
        top = 0
    bottom = int(pt[1] + size[1] / 2)
    d = len(image.shape)
    if d <= 2:
        #grayscale
        return image[top:bottom, left:right]
    else:
        #color
        return image[top:bottom, left:right, :]


def rot(image, degree):
    """
    画像を画像中央を中心に回転させる。回転により生じる背景は白で塗りつぶし。

    Parameters
    ----------
    image : mat
        元画像
    degree : float
        回転角度(deg)

    Returns
    -------
    回転した画像
    """
    (h, w) = (0, 0)
    if len(image.shape):
        #gray scale
        (h, w) = image.shape
        bg = (255, 255, 255)  # bg = 255
    else:
        #color
        (h, w) = image.shape[:2]
        bg = (255, 255, 255)
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, degree, scale = 1.0)  #回転のための変換行列を生成
    return cv2.warpAffine(image, mat, (w, h), borderValue = bg)     #アフィン変換


def mask_circle(image, dia):
    """
    円形でマスク。

    Parameters
    ----------
    image : mat
        画像
    dia : int
        円の直径(px)

    Returns
    -------
    マスクした画像
    """
    #マスク画像の元を作成（白一面）
    mask = np.full(image.shape[:2], 255, dtype = image.dtype)

    #白い画像の中心に黒い円を描画する。
    centerX = image.shape[1] // 2
    centerY = image.shape[0] // 2
    cv2.circle(mask, (centerX, centerY), (dia // 2), color = (0,0,0), thickness = - 1)
    #cv2.imshow('TMask', mask)
    #cv2.waitKey(0)
    return cv2.bitwise_or(image, mask)


def winShow(image, name, pt, size):
    """
    画像をウインドウで表示
    Parameters
    ----------
    image : mat
        画像データ
    name : string
        ウインドウ名
    pt : [number, number]
        ウインドウ表示位置[col, row]pixel
    size : [number, number]
        ウインドウに表示する画像サイズ[width, height]pixel
    Returns
    -------
    なし
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.moveWindow(name, pt[0], pt[1])
    cv2.resizeWindow(name, size[0], size[1])
    return


########  main  ######################################################################################
#指定したフォルダの画像すべてに対して実行
#pythonの関数の定義位置は、実際に実行するまでに定義されていればよい。
#プログラムは先頭から実行されていくがdef関数定義の中身は実行されないので、実際に呼び出された時で考える。
#ここでprocess()が呼ばれるまでにprocess()中で呼ばれる関数は定義されているのでprocess()の後に定義でもオッケー
print()
print()
print("*********************************************************")
print("* ホップ回転数測定                                      *")
print("*       Open CV による画像解析                          *")
print("*********************************************************")

print()

root = './bbpict'
resultdir = './bbresult/'
rootFiles = os.listdir(path = root)
rootFiles.remove('.DS_Store')
files = []
#ディレクトリを除外
for f in rootFiles:
    if os.path.isfile(root+'/'+f) is True:
        files.append(f)

files = sorted(files)
if len(files) == 0:
    print("画像ファイルがないので終了します。")
    sys.exit()

startImgFileName = os.path.splitext(os.path.basename(files[0]))[0]    #拡張子無しファイルネーム
endImgFileName = os.path.splitext(os.path.basename(files[-1]))[0]
csvFileName = resultdir+'result'+startImgFileName+'-'+endImgFileName+'_hoprot.csv'

print(f"画像 {startImgFileName} - {endImgFileName}  {len(files):3d}枚")

##日付時刻の取得してファイル名を作成
#t_delta = datetime.timedelta(hours=9)
#JST = datetime.timezone(t_delta, 'JST')
#now = datetime.datetime.now(JST)
# YYMMDDhhmmss形式に書式化
#dTime = now.strftime('%y%m%d%H%M%S')
#csvFileName = "rotdata" + dTime + ".csv"

#ヘッダー書き込み
with open(csvFileName, 'w', encoding="utf-8") as fCsv:     #'w'->上書き
    writer = csv.writer(fCsv)
    writer.writerow(['画像番号', '傾斜角度', '初速', 'センサ時間', 'ホップ回転数', '検出状態'])
    writer.writerow(['', '°', 'm/sec', 'usec', 'rps', ''])

### メインループ ########################################################
for i, f in enumerate(files):
    openFileName = os.path.join(root, f)
    print()
    print('-------------------------------------------------------------------------------------------------------')
    print()
    print(f' {i + 1:3d} 枚目 / {len(files):3d} 枚')
    print()
    print('filename = ', openFileName)
    resImg, incAngle, v0, dt, bbRot, statusE  = process(openFileName, '360standard')
    if statusE =='180redo':
        #広範囲角度解析で不具合が出ている時-180〜180度でやり直し
        print()
        print('----- 再計算1回目 角度制限-180〜180° --------------------------------------------')
        resImg, incAngle, v0, dt, bbRot, statusE  = process(openFileName, '180redo')
        if statusE =='90redo':
            print()
            print('----- 再計算2回目 角度制限-90〜90° ----------------------------------------------')
            #広範囲角度解析で不具合が出ている時-90〜90度でやり直し
            resImg, incAngle, v0, dt, bbRot, statusE  = process(openFileName, '90redo') ##すこしは救われるみたい。低回転側は無理な感じ

    ##画像をセーブ
    savedFileName = resultdir+"result"+f
    cv2.imwrite(savedFileName, resImg)

    ##測定値をcsvでセーブ
    #数値の整形
    txtIa = f"{incAngle:7.3f}"
    txtV0 = f"{v0:7.2f}"
    txtDt = f"{dt:7.2f}"
    if isinstance(bbRot, str):
        txtBbr = ' ---  '
    else:
        txtBbr = f"{bbRot:6.1f}"

    with open(csvFileName, 'a', encoding="utf-8") as fCsv:     #'a'->アペンド
        writer = csv.writer(fCsv)
        writer.writerow([os.path.splitext(os.path.basename(f))[0], txtIa, txtV0, txtDt, txtBbr, statusE])

    #################
    cv2.waitKey(3000)
    ##################
    #input('Returnキーを押す -> 次の画像へ')    ##自動処理

print()
print("-----------------------------------------------------------------------------------------------")
print("解析画像を.JPGファイルにセーブしました。")
print("回転アニメを.gifァイルにセーブしました。")
print("測定データ一覧表を.csvファイルにセーブしました。")
print('Complete')
print()
input('Push any key -> exit system ')
sys.exit()

