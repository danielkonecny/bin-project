1. Je naprogramování autoenkodéru čistě na nás? Jediná podmínka je využití obrázků jako datasetu, ale konkrétní volba
   datasetu je na nás? Počítám, že bych mohl projekt prvně otestovat na nějakém jednodušším datasetu a poté přejít k
   obrázkům s větším rozlišení.

> Autoenkodér nemusíte programovat sami, využijte k tomu TensorFlow a autoenkodér sestavte z prostředků, které Vám TensorFlow nabízí.
> Není podmínkou, že se musí jednat o obrázky, autoenkodér lze využít na různorodou škálu problémů, nicméně obrázky jsou asi nejvíce intuitivní.
> Jelikož je součástí projektu i statistické vyhodnocení, které vyžaduje spuštění několika běhů, tak by bylo vhodné použít nějaký menší dataset (stačí MNIST, fashion-MNIST nebo něco obdobně "malého"), jinak budete mít problém s dobou výpočtu.

2. Máme využít nějakého algoritmu na porovnání kvality vstupního s výstupním obrázkem, abychom mohli nějak číselně
   reprezentovat kvalitu obrazu za dané komprese? Tedy by výstupem projektu mohlo být například: za dané komprese má
   dekompresovaný obrázek následující ztrátu kvality, pokud využijeme menší komprese, dosáhneme téměř originální kvality
   obrázku?

> K porovnání podobnosti dvou obrázků existuje několik metod, nejjednodušší je asi spočítat MSE (Mean Square Error) další možností je operace konvoluce/korelace a další.

3. Má evoluční algoritmus zkoušet zásadně měnit architekturu autoenkodéru, tedy například měnit počet konvolučních
   vrstev? Či má za cíl spíše najít ideální parametry jednotlivých vrstev, např. počet uzlů dense vrstvy, počet kernelů
   u konvoluční vrstvy, apod.?

> Obě možností jsou ve hře, můžete buď zkoušet měnit jen parametry, nebo celou architekturu anebo obojí.

4. Ještě jsme neměli přednášku o evolučních algoritmech, takže vycházím čistě ze znalostí z předchozích předmětů (IZU,
   SFC), to jen pro případ, kdy by odpověď na mou následující otázku měla být na některé z přednášek. Je nějaký
   konkrétní evoluční algoritmus, který máme využít, či který je vhodný na tento problém, nebo je volba čistě na nás?

> Volba je na Vás, můžete se klidně inspirovat nějakým algoritmem na internetu. To, zda se daný algoritmus hodí na řešený problém by právě měly ukázat ty statistické testy.

5. Můžeme využít nějakou existující implementaci evolučního algoritmu, nebo si máme evoluční algoritmus naprogramovat od
   píky sami?

> V tomto předmětu se počítá, že daný evoluční algoritmus implementujete Vy, pokud by se ale jednalo o nějaký složitější algoritmus, tak je možné využít nějaké již existující implementace jako základ pro Váš algoritmus. Důležité je, abyste jen nepoužil něco co už existuje bez nějaké Vaší přidané hodnoty. V tom evolučním algoritmu musí být něco co jste udělal Vy. Například je možné pro výpočet pareto fronty (možná už znáte, ale bude se to probírat) použít [toto](https://github.com/ehw-fit/py-paretoarchive).

> V projektu byste se měl hlavně zaměřit na ten evoluční algoritmus a to statistické vyhodnocení, ten autoenkodér berte spíše jako problém, který se pomocí toho evolučního algoritmu snažíte vyřešit.

> Například: na internetu najdete nějaký problém, který lze pomocí autoenkodéru řešit (třeba rekonstrukce ručně psaných čísel, jako je v tutorialu na tensorflow nebo odstranění šumu z obrázku). V tutorialu najdete nějakou architekturu autoenkodéru s X vrstvami a Y parametry, která převádí obrázek do kompresované formy velikosti MxN. No jenomže takovýto model je jen jedním z mnoha a může existovat jiný model, který má méně vrstev a/nebo méně parametrů a/nebo převádí obrázek ještě do menší formy při zachování stejné (nebo jen o něco menší) vykonnosti sítě. No a Váš evoluční algoritmus by se měl snažit takovýto model najít.
