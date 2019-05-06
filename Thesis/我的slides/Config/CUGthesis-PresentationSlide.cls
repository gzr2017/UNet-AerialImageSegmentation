% !TEX encoding = UTF-8
%%
%% This file `CUGthesis-PresentationSlide.cls', which is written by YangYuancong CUG based on ZhouWei's PHD Thesis ,Ocean University of China,
%% The First Edition of  The File is
%% is the configuration file of the CUGthesis-PresentationSlide  package with LaTeX2e.
%%
%% Special thanks to Dr.HuangZhnghua,Wuhan University  and Dr.ZhouWei，Ocean University of China
%%
%% This file was generated from file(s) of the ZWPresentationSlide package project.  http://blog.sciencenet.cn/home.php?mod=space&uid=453771&do=blog&id=456252
%% ----------------------------------------------------------------------
%%
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%%
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{Config/CUGthesis-PresentationSlide}
  [2013/05/28 v1 CUGthesis-PresentationSlide  document class]
  \newif\ifwinfonts\winfontstrue                                                      %%%%%%%字体选择,在选项中未指明字体时优先使用winfonts，因此将之设置为true 关于 \newif 查看 http://handyfloss.wordpress.com/2007/08/29/latex-programming-how-to-implement-conditionals/
 \newif\ifadobefonts\adobefontsfalse
\DeclareOption{winfonts}{\winfontstrue  \adobefontsfalse}                     %如果选项中确定使用 winfonts 则将ifwinfonts设为true、\adobefontsfalse
\DeclareOption{adobefonts}{\adobefontstrue \winfontsfalse}                 %如果已选择winfonts 那么要将adobefonts设为false,假设不注意同时使用了winfonts和adobefonts，那么优先选用的是adobefonts，这种情况下如果想优先选用adobefonts，则交换本行和上一行
%%接受从documentclass传入的参数 然后加载beamer文档类
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}%重要，此处接受从documentclass传入的参数到基础类中，没有此行和下面一行则documentclass传入的参数无效
\ProcessOptions
\LoadClass[notheorems,mathserif,compress]{beamer} 
%--------------------------------------------------------------------------以下加载宏包--------------------------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{etex}  %使用etex解除tex的256限制   http://ctan.org/pkg/etex/
\RequirePackage{ifpdf,ifxetex}
%%注意, beamer 会默认使用下列宏包: amsthm, graphicx, hyperref, color, xcolor,这里显式写出
%\RequirePackage{amsmath,amsthm,amsfonts,bm,txfonts}
\RequirePackage{amsmath,amsthm,amsfonts,txfonts}
\RequirePackage[fleqn,tbtags]{mathtools}	%参看http://ftp.ctex.org/mirrors/CTAN/macros/latex/contrib/mh/mathtools.pdf  http://blog.sina.com.cn/s/blog_5e16f1770100obgq.html

\RequirePackage{amssymb}			%http://www.tex.ac.uk/cgi-bin/texfaq2html?label=manymathalph
  \newcommand{\hmmax}{0}		%解决 No room for a new \mathgroup
  \newcommand{\bmmax}{0}
\RequirePackage{bm}
%\RequirePackage[small,euler-digits]{eulervm} %数学字体 http://bbs.ctex.org/forum.php?mod=viewthread&tid=43260
\RequirePackage{graphicx, color}
\RequirePackage{hyperref}
        \hypersetup{CJKbookmarks,%
       bookmarksnumbered,%
              colorlinks,%
               linkcolor=blue,%
               citecolor=blue,%
              plainpages=false,%
            pdfstartview=FitH}
\RequirePackage{xcolor} %%allows you to use names of the default  300 colors  http://en.wikibooks.org/wiki/LaTeX/Colors but any of these
			%%Options [usenames,dvipsnames,svgnames,table] can't be used here , if any of these be used ,the error message is "Option clash for package xcolor."
	%%%%%%%%define user’s custom color Microsoft Pink 2013
	\definecolor{MicrosoftPink}{RGB}{220,79,173}
	\definecolor{MicrosoftCrimson}{RGB}{172,25,61}	
	\definecolor{MicrosoftDeepOrange}{RGB}{210,71,38}
	\definecolor{MicrosoftOrange}{RGB}{255,143,50}
	\definecolor{MicrosoftLightGreen}{RGB}{130,186,0}
	\definecolor{MicrosoftGreen}{RGB}{0,138,23}
	\definecolor{MicrosoftLightCyan}{RGB}{3,179,178}	
	\definecolor{MicrosoftCyan}{RGB}{0,130,153}
	\definecolor{MicrosoftLightBlue}{RGB}{93,178,255}
	\definecolor{MicrosoftBlue}{RGB}{0,114,198}		 %%Microsoft‘s Blue in RGB Mode  ，for more details visit http://en.wikibooks.org/wiki/LaTeX/Colors
	\definecolor{MicrosoftDeepPurple}{RGB}{70,23,180}
	\definecolor{MicrosoftPurple}{RGB}{140,0,149}	
	\definecolor{MicrosoftMediumDark}{RGB}{0,75,139}
	\definecolor{MicrosoftNavy}{RGB}{0,25,64}
	\definecolor{MicrosoftBrown}{RGB}{87,0,0}
	\definecolor{MicrosoftDarkBrown}{RGB}{56,0,0}
	\definecolor{MicrosoftMediumGray}{RGB}{88,88,88}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%图形 表格 多媒体
\RequirePackage{subfigure} %%图形或表格并排排列
\RequirePackage{xmpmulti}  %%支持文中的 \multiinclude 等命令, 使 mp 文件逐帧出现. 具体讨论见 beamer 手册.
\RequirePackage{tikz,tikz-3dplot,pgfplots}                       %以前采用的是pgf,  tikz 是pgf的更高层，易用,因此在此处采用
                                                                                %   http://mixing.coas.oregonstate.edu/links/latex_files/pgfuserguide.pdf
        \pgfdeclareimage[  % height=1.3496cm,      % 可以启用主文件中的\input{Sections/Silder_DisplayTheFrameLayout.tex}，获得单帧的布局情况，计算宽高，不过更推荐width=\paperwidth
                                        %width=14cm,                 %图片的宽度(根据宽度来自动缩放)
                                        width=\paperwidth,         %使图片为整个页面宽度自动缩放(推荐)
                                        interpolate=true ,          %按比例缩放到指定宽度或高度值
                                    ]{CUGLogoHome}{PictureData/CUGLogo/CUGLogoHome} %不需要往上级目录(即路径不要写成这样 ../PictureData/CUGLogo/CUGLogoHome  直接用 PictureData/CUGLogo/CUGLogoHome)，调用该类时其所在目录为\documentclass所在目录。
        %正文页中显示的校徽，一并定义在此处，原来定义在170行左右
        %\logo{\includegraphics[height=1.7cm,width=1.7cm, bb=0 0 700 73]{PictureData/CUGLogo/CUGLogo}}%% 校徽,最好是 MetaPost 文件. 正文中左上角一直显示
        \logo{\includegraphics[width=1.6197cm]{PictureData/CUGLogo/CUGLogo}}%% 校徽,最好是 MetaPost 文件. 正文中左上角一直显示
\usetikzlibrary{arrows,shapes,positioning,calc,fadings,decorations.pathreplacing}  %TikZ 各种库
\RequirePackage{verbatim}
%%%-------表格
\RequirePackage{colortbl,dcolumn}     %% 彩色表格
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{makecell}
%%%%%%%%%%%%%%%%%%其他
\RequirePackage[sort&compress,numbers]{natbib}
\RequirePackage{hypernat}
\RequirePackage{layout}
\RequirePackage{times}      %%Times New Roman字体包，将英文字体默设为Times New Roman
\RequirePackage{textcomp}  %摄氏度之类的符号
\RequirePackage{booktabs}
\RequirePackage{pifont,bbding}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%脚注%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{pifont,bbding}
\RequirePackage[%perpage, %为每页脚注单独排序
		stable,	 %避免章节标题中的脚注随同章节标题出现在目录或页眉之中
		multiple, %给正文中两个以上的并排脚注标号之间加上分隔逗号
		symbol*	  %将脚注的数字序号改为 * 号等不同的符号
		]{footmisc}
\DefineFNsymbols{lamport-mystyle}{\SnowflakeChevron \OrnamentDiamondSolid  \FourClowerOpen \FourStarOpen \SixFlowerPetalDotted \JackStarBold  \textparagraph \textdagger \textdaggerdbl\PencilLeft \NibSolidLeft \Cross \CrossClowerTips \CrossMaltese \Asterisk\FiveFlowerPetal \FiveStar  \AsteriskCenterOpen \AsteriskRoundedEnds  \SixFlowerPetalRemoved \SixFlowerRemovedOpenPetal\EightAsterisk \EightFlowerPetalRemoved \Snowflake  \S * {**}}
\setfnsymbol{lamport-mystyle}
%if Specified a symbol which must be used in math enviroment ,then chang "symbol*" to "symbol" and use the define \DefineFNsymbols{lamport}[math]{....symbol....}  ---\DefineFNsymbols{lamport}[math]{*\dagger\ddagger\S\P\|{**}{\dagger\dagger}{\ddagger\ddagger}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%脚注%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{diagbox}	%斜线表格
\RequirePackage[figuresright]{rotating}
\RequirePackage{tagging}  %条件编译

\RequirePackage{array}                %定义横粗线\while 和纵粗线I
\newcolumntype{I}{!{\vrule width 2pt}}
\newlength\savewidth
\newcommand\shline{\noalign{\global\savewidth\arrayrulewidth
                            \global\arrayrulewidth 2pt}%
                   \hline
                   \noalign{\global\arrayrulewidth\savewidth}}
%以上为横纵粗线定义
%设定表格宽度
\newcommand{\PreserveBackslash}[1]{\let\temp=\\#1\let\\=\temp}
\newcolumntype{C}[1]{>{\PreserveBackslash\centering}p{#1}}

%%代码高亮
\RequirePackage{listings}
\lstset{
  language=Matlab,
  numberbychapter=true,
  basicstyle=\small,
  numbers=left,
  keywordstyle=\color{blue},
  numberstyle={\tiny\color{lightgray}},
  stepnumber=1,                      %行号会逐行往上递增
  numbersep=5pt,
  commentstyle=\small\color{red},
  backgroundcolor=\color[rgb]{0.95,1.0,1.0},
  showspaces=false,                  %显示空格
  showtabs=false,
  breaklines=true,                   %自动换行
  frame=shadowbox, framexleftmargin=5mm, rulesepcolor=\color{red!20!green!20!blue!20!},
    flexiblecolumns=true,
    breaklines=true,            %对过长的代码自动换行
    breakautoindent=true,
    breakindent=1em,
   keepspaces=true,
    extendedchars=false,
    columns=flexible,
    mathescape=true,
     frame=single,
%  TABframe=single,
  tabsize=4,
  aboveskip=1em,
  breaklines=tr,
  extendedchars=false %这一条命令可以解决代码跨页时，章节标题，页眉等汉字不显示的问题
}
%以上代码宏包
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------以上加载宏包--------------------------------------------------------------------------------------------------------------%

%\logo{\includegraphics[height=1.7cm,width=1.7cm, bb=0 0 700 73]{PictureData/CUGLogo/CUGLogo}}%% 校徽, MetaPost 文件. 左上角一直显示
%\logo{\includegraphics[width=1.3cm]{PictureData/CUGLogo/CUGLogo}}%% 校徽, MetaPost 文件. 左上角一直显示

\renewcommand{\raggedright}{\leftskip=0pt \rightskip=0pt plus 0cm}
\raggedright %% 中文对齐

\def\hilite<#1>{\temporal<#1>{\color{blue!35}}{\color{magenta}}{\color{blue!75}}}
%% 自定义命令, 源自 beamer_guide. item 逐步显示时, 使已经出现的item、正在显示的item、将要出现的item 呈现不同颜色.

\newcolumntype{H}{>{\columncolor{blue!20}}c!{\vrule}}
\newcolumntype{H}{>{\columncolor{blue!20}}c}  %% 表格设置
%==================================参考文献==============================================================
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}  %自定义命令\upcite, 使参考文献引用以上标出现
\bibliographystyle{plain}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 模式判断
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifxetex                             %Linux下xelatex使用xeCJK环境(虽然看起来像是重复了ctex宏包的工作，但如果直接调ctex宏包，却有更多的问题要解决)，xetex部分开始      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  \RequirePackage[CJKnumber=true,		%是否使用 CJKnumb 宏包
		   indentfirst=true,	%在章节标题后面的第一段首行是否缩进
		   CJKmath=true,	%是否支持在数学环境中直接输入 CJK 字符
                   CheckSingle=true,	%是否避免单个 CJK 文字单独占一个段落的最后一行
                   AutoFallBack=true,	%当文档中有个别生僻字时,可以使用这个选项,自动使用预先设置好的后备字体来输出这些生僻字
		   PunctStyle=kaiming,   %设置标点处理格式
		   boldfont,
		   slantfont,
		   CJKecglue=\hskip 0pt plus 0.08\baselineskip minus 2pt  %设置 CJK 文字与西文、 文字与行内数学公式之间的间距, 此处设为与CJKglue一样  http://blog.csdn.net/ljfrog/article/details/8820637
		]{xeCJK}    % http://ctan.mirrorcatalogs.com/macros/xetex/latex/xecjk/xeCJK.pdf
  \DeclareGraphicsExtensions{.pdf,.png,.jpg,.mps}
  \DeclareGraphicsRule{*}{eps}{*}{} %% 使 xelatex 可以纳入 metapost 做的图片.
  %%%%%%%%%%%%%%%%%%%%%%%%%%    xeCJK字体设置
        \ifwinfonts
                              %windows字体，来自ctex的 ctex-xecjk-winfonts.def
             \setCJKmainfont[BoldFont={SimHei},ItalicFont={KaiTi}] {KaiTi}
             \setCJKsansfont{SimHei}
             \setCJKmonofont{FangSong}

             \setCJKfamilyfont{zhsong}{SimSun}
             \setCJKfamilyfont{zhhei}{SimHei}
            \setCJKfamilyfont{zhkai}{KaiTi}
             \setCJKfamilyfont{zhfs}{FangSong}
             \setCJKfamilyfont{zhli}{LiSu}
             \setCJKfamilyfont{zhyou}{YouYuan}
             \newcommand{\songti}{\CJKfamily{zhsong}} % 宋体
             \newcommand{\heiti}{\CJKfamily{zhhei}}   % 黑体
             \newcommand{\kaishu}{\CJKfamily{zhkai}}  % 楷书
             \newcommand{\fangsong}{\CJKfamily{zhfs}} % 仿宋
             \newcommand{\lishu}{\CJKfamily{zhli}}    % 隶书
             \newcommand{\youyuan}{\CJKfamily{zhyou}} % 幼圆
      \fi
     \ifadobefonts %adobe字体，来自ctex的 ctex-xecjk-adobefonts.def
                \setCJKmainfont[BoldFont=Adobe Heiti Std,ItalicFont=Adobe Kaiti Std] {Adobe Fangsong Std}
                \setCJKsansfont{Adobe Heiti Std}
                \setCJKmonofont{Adobe Fangsong Std}

                 \setCJKfamilyfont{zhsong}{Adobe Song Std}
                 \setCJKfamilyfont{zhhei}{Adobe Heiti Std}
                 \setCJKfamilyfont{zhfs}{Adobe Fangsong Std}
                 \setCJKfamilyfont{zhkai}{Adobe Kaiti Std}
                 \setCJKfamilyfont{zhli}{LiSu}                      % adobe 没有隶书和幼圆，因此用windows的代替
                 \setCJKfamilyfont{zhyou}{YouYuan}

                 \newcommand{\songti}{\CJKfamily{zhsong}} % 宋体
                 \newcommand{\heiti}{\CJKfamily{zhhei}}   % 黑体
                 \newcommand{\kaishu}{\CJKfamily{zhkai}}  % 楷书
                 \newcommand{\fangsong}{\CJKfamily{zhfs}} % 仿宋
                 \newcommand{\lishu}{\CJKfamily{zhli}}    % adobe 没有隶书和幼圆，因此用windows的代替
                 \newcommand{\youyuan}{\CJKfamily{zhyou}} % 幼圆
       \fi
\else                                                      %xetex部分结束      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  \ifpdf                                                   %pdftex部分开始      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      \RequirePackage{CJK,CJKnumb}    %Windows 下的pdflatex使用CJK环境
      \RequirePackage{ccmap}          %%使PDF的文字可以正常复制
    \DeclareGraphicsExtensions{.pdf,.png,.jpg,.mps}
    \DeclareGraphicsRule{*}{mps}{*}{} %% 使 pdflatex 可以纳入 metapost 做的图片.
    %%%%%%%%%重定义字体
    \newcommand{\songti}{\CJKfamily{song}}        % 宋体
    \newcommand{\fangsong}{\CJKfamily{fs}}        % 仿宋体
    \newcommand{\kaishu}{\CJKfamily{kai}}         % 楷体
    \newcommand{\heiti}{\CJKfamily{hei}}          % 黑体
    \newcommand{\lishu}{\CJKfamily{li}}           % 隶书
   \newcommand{\youyuang}{\CJKfamily{you}}       % 幼圆
  \fi
\fi                                                          %pdftex部分结束      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%字号命令 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\sihao}{\fontsize{14pt}{\baselineskip}\selectfont}      % 字号设置 4号
\newcommand{\xiaosihao}{\fontsize{12pt}{\baselineskip}\selectfont}  % 字号设置 小4号
\newcommand{\wuhao}{\fontsize{10.5pt}{\baselineskip}\selectfont}    % 字号设置 5号
\newcommand{\xiaowuhao}{\fontsize{9pt}{\baselineskip}\selectfont}   % 字号设置 小5号
\newcommand{\liuhao}{\fontsize{7.875pt}{\baselineskip}\selectfont}  % 字号设置 6号
\newcommand{\qihao}{\fontsize{5.25pt}{\baselineskip}\selectfont}    % 字号设置 7号
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%主题定义
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%个人定义%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\useoutertheme[height=0.1\textwidth,width=0.15\textwidth,hideothersubsections]{sidebar}
\usecolortheme{whale}      % Outer color themes, 其他选择: whale, seahorse, dolphin . 换一个编译看看有什么不同.
\usecolortheme{orchid}     % Inner color themes, 其他选择: lily, orchid
\usefonttheme[onlysmall]{structurebold}
\usefonttheme[onlymath]{serif}		%更漂亮的数学字体  http://sealhuang.sinaapp.com/?p=9
\usefonttheme{professionalfonts}        %更漂亮的数学字体  http://bbs.ctex.org/forum.php?mod=viewthread&tid=71614
\useinnertheme[shadow]{rounded} % 对 box 的设置: 圆角、有阴影.
%\setbeamercolor{sidebar}{bg=MicrosoftDeepPurple} % sidebar的颜色, 原来为50%的蓝色.
%\setbeamercolor{background canvas}{bg=blue!9} % 背景色, 9%的蓝色. 去掉下一行, 试一试这个.
\setbeamertemplate{background canvas}[vertical shading][bottom=white,top=MicrosoftBlue] %%背景色, 上MicrosoftBlue(定义在), 过渡到下白.
%\usefonttheme{serif}  % 字体. 个人偏好有轮廓的字体. 去掉这个设置编译, 就看到不同了.
%\setbeamertemplate{navigation symbols}{}   %% 去掉页面下方默认的导航条.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%简化复杂的CJK环境%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{etoolbox}       %% 该宏包提供比\AtBeginDocument 更优秀的document环境个性化方法 具体参看  http://mirrors.med.harvard.edu/ctan/macros/latex/contrib/etoolbox/etoolbox.pdf
\AfterEndPreamble{      %%%在\begin{document}后需要添加的内容
                                    %%为了简化环境，采用这种方式，具体参考 http://tex.stackexchange.com/questions/14135/how-to-automatically-add-text-immediately-after-begindocument
\ifpdf     %若是pdftex则要加入 CJK环境，xetex使用的xeCJK则不需要
   \begin{CJK*}{GBK}{kai}              %加入此环境是为了定理、证明......等的中文化参看  http://math.ecnu.edu.cn/~jypan/Teaching/MathSoft/Latex/lect10n.pdf
\fi
%%----------------------- Theorems ---------------------------------------------------------------------
\newtheorem{theorem}{定理}
\newtheorem{definition}{定义}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{proposition}{命题}
\newtheorem{example}{例}
\newtheorem{remark}{注}

\renewcommand\figurename{\rm 图}
\renewcommand\tablename{\bf 表}

\beamertemplateballitem %使itemize环境中的变成小球，这是一种视觉效果
%\beamertemplatenumberedballsectiontoc %为所有已编号的部分设置一个章节目录，并且编号显示成小球
%\beamertemplateboldpartpage %将每一页的要素名设成加粗字体


\AtBeginSection[] % Do nothing for \section*  %每节前加入目录页
{
   \transdissolve
   \begin{frame}<beamer>
  % \frametitle{Outline}
%   \tableofcontents[currentsection,hideallsubsections]
   \tableofcontents[sectionstyle=show/shaded,subsectionstyle=show/shaded/hide]
   \end{frame}
}

\setbeamertemplate{caption}[numbered] %图表编号
%%----------------------------------------------------------------------------------------------------
}

\AtEndDocument{       %%%在\end{document}前需要添加的内容
\ifpdf        %若是pdftex则要加入 CJK环境,xetex使用的xeCJK则不需要
 \end{CJK*}
 \fi
 }

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%User's Command%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%幻灯片首页
\newcommand{\MakeFrontCover}{

\begin{frame}[plain]
    %%幻灯片第一页图像CUGLogoHome使用 \pgfdeclareimage定义在 Config/CUGthesis-PresentationSlide第81行
 \ifpdf             %由于在Linux下和windows下首页的logo放置位置稍有不同，加上下面的条件判断
    \pgfputat{\pgfxy(-2.120,-0.60272)}{\pgfbox[left,base]{\pgfuseimage{CUGLogoHome}}}  %启用主文件中的\input{Sections/Silder_DisplayTheFrameLayout.tex}，获得一帧的布局，①、 ②的值分别对应坐标(x,y)(两条虚线的交点)为当前页面的坐标原点，y为上正下负，x右正左负
 \fi                                                                                                                                                %\pgfputat放置图片的左下角将基于“原点”，通过观察latout显示的信息可以精确的放置图片，假如当前两条虚线的交点距页面的左边界和上边界分别为2cm和3cm，如果图片要紧靠左上角
     \ifxetex                                                                                                                                   %则\pgfxy的参数应为(-2cm,height-3cm) height为图片高度，这样图片就能紧靠左边界和上边界放置了，这样的计算并不准确只能做参考
   \pgfputat{\pgfxy(-2.5441,1.666445)}{\pgfbox[left,base]{\pgfuseimage{CUGLogoHome}}}  %在本演示文稿中，已经设定宽度为整个页面宽度，虚线交点为(1inch,1inch),作为参考，一点一点的试，直到找到最合适的值，单位为cm  各单位换算关系访问 http://en.wikibooks.org/wiki/LaTeX/Lengths
       \fi                                                                                                                                          %针对本文件(CUGLogoHome.pdf)默认页面大小128mm × 96mm在标题(titile)为两行显示是的最合适值为(-2.120,-0.60272),经几十次修改的最佳值，你也可也将图片保存为当前CUGLogoHome.pdf的分辨率(982×106)，这样就不必更改此处参数
   \maketitle
   \end{frame}
      %\frame{ \titlepage }
}


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%User's Command%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ExtraConfig%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %\beamertemplatetransparentcoveredhigh  %使所有被隐藏的文本高度透明
\setbeamercovered{transparent}      %半透明化尚未出现的内容,方便提醒自己下面要讲什么
