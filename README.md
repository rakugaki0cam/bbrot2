# bbrot

オリジナルは60bbaaさんのbbrot

https://github.com/60bbaa/bbrot


うちの撮影条件に合うように少々改変しております。

目的:bb弾の回転角度を画像より読み取り、ホップ回転数を求めます。

原理:
多重撮影された複数のBB弾をブロブ検出して、中央付近の1コマをテンプレートとします。
テンプレートを1度ずつ回転させ、マッチ度が最大の時の角度を読み取り角度とします。

撮影ストロボ発光周期はBBが12mmすすむ時間で、初速検出部にて測定しています。
発光周期の値はLCDに表示しているものを
今のところ、OCRにて正確に値を読み取れないため手入力としています。

ホップ回転数[rps]　= 1000000usec / (発光周期[usec] x コマ数) x (回転角[°] / 360°)


