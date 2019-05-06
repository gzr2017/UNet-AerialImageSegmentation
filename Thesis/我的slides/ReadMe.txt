各文件关系请查看 YangYuancong'sPresentationSlide_各文件间的主要关系.png  或访问http://www.gliffy.com/go/publish/image/4671030/L.png


	Windows用户 ：保证 CTeX完整版以默认方式安装，PDFLaTex编译两次；已知问题：若在Winedit中直接编译PDFLaTeX会因为编码不支持问题出现书签为乱码的情况，解决方法是直接双击运行Make.bat。编译完成后用AbobeReader 或AdobeAcrobat全屏放映即可看到效果
	Linuxer     : 安装Texlive，make或make Realse，具体查看YangYuancong'sPaper\ReaderMe__Linuxer.txt，编译完成使用evince放映即可看到效果

推荐入门(先学习latex):
	http://zoho.is-programmer.com/user_files/zoho/File/beamerlog-1112.pdf
	http://ctex.blogbus.com/logs/53001706.html
	
	当然最好的还是: http://faq.ktug.org/wiki/uploads/beamer_guide.pdf

比较全面的幻灯片模板:http://ishare.iask.sina.com.cn/f/22664641.html
一个德国人做的非常适合学术使用的主题 http://www.cgogolin.de/Computer.html


建议：了解一下xeCJK，会有很大帮助 http://ctan.mirrorcatalogs.com/macros/xetex/latex/xecjk/xeCJK.pdf

学习xeCJK：http://hi.baidu.com/ennmoziakkmpxye/item/5079a1c629de854ca9ba94b8
	   http://forum.ubuntu.org.cn/viewtopic.php?f=35&t=247476&start=0
			

  使用图片时最好不要加扩展名，增加自由度，LaTeX 会自动按照.png->.pdf->.jpg->.mps->.jpeg->.jbig2->.jb2->.PNG->.PDF->.JPG->.JPEG->.JBIG2->.JB2 的优先级顺序来查找文件,也不要带路径，使用\graphicspath{{path1}{path2}{...}}来告诉latex该在哪些地方找图片



说明：
	1、本幻灯片主题及设置基于中国海洋大学2011年博士论文幻灯片答辩模板，
		http://blog.sciencenet.cn/home.php?mod=space&uid=453771&do=blog&id=456252
		其最原始出处为 武汉大学 数学与统计学院 信息与计算科学系 黄正华老师http://aff.whu.edu.cn/huangzh/    一个用 beamer 做幻灯片的例子

	2、Makefile文件基于上海交大beamer模板  https://github.com/X-Wei/aBeamerTemplate4SJTU
		Windows用户请无视

	3、如果是CTex环境，请使用CTex套装	http://www.ctex.org/CTeXDownload/


	4、Config/XcolorDefined.psd  和Config/XcolorDefined.pdf 标识的是 CUGthesis-PresentationSlide.cls 中定义的各种xcolor 的RGB 形式的颜色效果，这些颜色均来Microsoft	   		的outlook邮箱主题颜色，因此在所有颜色名前冠以Microsoft，比如XcolorDefined.pdf 的Blue，其在 CUGthesis-PresentationSlide.cls中的定义名为MicrosoftBlue

	5、PictureData\CUGLogo  目录下的CUGLogo.png 分辨率为1154×770， CUGLogoHome.pdf 的分辨率为982×106， 分别作为正文左上角上角Logo和演示文稿首页的横幅，该这两个分			辨率，对应 CUGthesis-PresentationSlide.cls定义的    \logo{\includegraphics[width=1.6197cm]{PictureData/CUGLogo/CUGLogo}}和
		    \pgfputat{\pgfxy(-2.120,-0.60272)}{\pgfbox[left,base]{\pgfuseimage{CUGLogoHome}}}中的width=1.6197cm 和 \pgfxy(-2.120,-0.60272)，是经过几十次试验的			结果，刚好能将这两张图片放到固定的位置而不超出边界一个像素或差一个像素才到边界，可以说是严丝合缝，因此，如果想更改自己的Logo，最好保存成这两个文件一样			的高宽比;当然如果你想试一试自己重新定义格式另当别论
	6、喜欢简洁不需要logo的，注释掉CUGthesis-PresentationSlide.cls 中 \logo 和 \pgfputat所在行，喜欢纯色背景的，CUGthesis-PresentationSlide.cls \setbeamercolor  		\setbeamertemplate 命令处有详细说明


日志:

	1、2013.5.26 从中国海洋大学2011年博士论文幻灯片答辩模板 ZWPresentationSlide 整理，不采用原模板的单个文件模式，而是分离各部分，抽取出一个CUGthesis-				PresentationSlide.cls, 分离各节到Sections目录下单独tex文件中，图片也保存到PictureData目录下去，显得结构更清晰
	2、2013.5.27 更换演示文稿首页Logo，确定图片分辨率982×106和pgfxy参数值(-2.120,-0.60272)，保证放置Logo完全准确
	3、2013.5.29 更换演示文稿正文左上角Logo，确定分辨率为1154×770和width为1.6197cm,保证放置完全准确
	4、2013.5.30 编写Makefile，其中包含自动转换编码
	5、2013.6.2 \title 中嵌套tabular表格，更好的对齐方式
	6、2013.6.11  加入etex宏包  http://ctan.org/pkg/etex/
	7、2013.6.12  更新文档类设定，windows 下pdflatex 使用 CJK环境，Linux下xelatex使用xeCJK环境，使用条件判断
	8、2013.6.13  加入幻灯片切换效果演示和PGF/TikZ 矢量图实例






=================================================================================
黄正华老师的主页
http://aff.whu.edu.cn/huangzh/

ctex hua's beamer page
http://bbs.ctex.org/viewthread.php?tid=27695


latex 编辑部 beamer
http://zzg34b.w3.c361.com/templet/slide.htm

于海军老师主页
http://dsec.pku.edu.cn/~yuhj/wiki/TeXSlides.html#sec-5