from keras._tf_keras.keras.models import Model, load_model
import numpy as np
import cv2
import sys
import os

WINDOW_TITLE	= 'MobileNetV2'
FONT_FACE	= cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE	= 2
FONT_THICKNESS	= 2
FONT_COLOR	= (255, 255, 255)
FONT_LINETYPE	= cv2.LINE_8
LABELS= (
	'pyramid-djoser',
	'pyramid-khafre',
	'pyramid-menkaure',
	'pyramid-senefru',
	'sculpture-hatshepsut',
	'sculpture-isis',
	'sculpture-nefertiti',
	'sculpture-ramesses-II',
	'statue-akhenaten',
	'statue-djoser',
	'statue-hatshepsut',
	'statue-horus',
	'statue-khafre',
	'statue-ramesses-II',
	'statue-sphinx',
	'statue-tuhotmus-III',
	'temple-abu-simbel',
	'tutankhamun',
)

def main(argv=sys.argv[1:]):
	argv.insert(0, sys.argv[0]) #prepend the program path to arguments
	argc= len(argv)
	if argc==2:
		fn= argv[1]
		if not os.path.exists(fn):
			print('Failed to find image "%s"'%(fn))
			return
		
		model:Model= load_model('mobilenetv2.keras')
		img= cv2.imread(fn)
		preds= model.predict(cv2.resize(img, (224, 224)).reshape((1, 224, 224, 3)))

		idx= np.argmax(preds[0])
		text= '%s (%f)'%(LABELS[idx], preds[0][idx])
		print(text)

		img= cv2.resize(img, (1366, 768))
		extents, baseline= cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
		cv2.putText(img, text, (0, extents[1]+baseline), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, FONT_LINETYPE)
		cv2.imshow(WINDOW_TITLE, img)
		cv2.waitKey(0)
	else:
		print('USAGE:\npython "%s" "path/to/image"'%(sys.argv[0]))

if __name__=='__main__':
	main()
