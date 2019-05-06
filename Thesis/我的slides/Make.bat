@echo on
::删除原pdf文件同时作备份源文件到当前用户的桌面,随时备份文件是很重要的
::参看http://blog.csdn.net/wzsbll/article/details/6690895
del *.pdf
::判断系统版本 只判断XP与Win7,根据返回值判断 http://blog.163.com/xqslove@yeah/blog/static/1670140142012112221429906/  http://www.myexception.cn/operating-system/726032.html
@ver|findstr "5.1"
::如果是Windows7 目标设为 Desktop
@if %errorlevel% equ 1 (set Dest=Desktop)
::如果是Windows XP  目标设为 桌面
@if %errorlevel% equ 0 (set Dest=桌面)
::del  /Q /S  "%HOMEDRIVE%%HOMEPATH%\%Dest%\PresentationSlide"
XCOPY  "*"  "%HOMEDRIVE%%HOMEPATH%\%Dest%\PresentationSlide" /E /Y /D
@pause
::exit

::设置主文件的文件名
set  Name=YangYuancong'sPresentationSlide

::第一次编译建立索引，为交叉引用作准备
pdflatex "%Name%.tex"
makeindex -o "%Name%.ind" "%Name%.idx"
::把书签编码从GBK转换为Unicode，不然书签乱码，最后一次完整索引编译
Config\GBK2Uni_Windows.exe "%Name%.out"
pdflatex "%Name%.tex"

::打开文件
start %Name%.pdf


Config\Cleaner.bat
@pause
