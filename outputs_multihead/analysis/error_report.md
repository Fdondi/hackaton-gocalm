# Predictions Error Analysis

- docs analyzed: **100**
- true positives: **145**
- merged-span true positives: **2**
- false positives: **8**
- false negatives: **2**
- partial overlaps (subset): **16**
- partial overlaps (superset): **4**
- type mismatches: **3**

## False Positives Per Category

| category | count |
| --- | --- |
| OTHER | 4 |
| ADDRESS | 2 |
| ORG | 2 |

## False Negatives Per Category

| category | count |
| --- | --- |
| OTHER | 2 |

## Partial Range Overlaps Per Category

### Subset Errors (predicted is narrower than gold)

By gold category:

| gold_category | count |
| --- | --- |
| ADDRESS | 12 |
| OTHER | 3 |
| PHONE | 1 |

By predicted category:

| predicted_category | count |
| --- | --- |
| ADDRESS | 12 |
| OTHER | 3 |
| PHONE | 1 |

### Superset Errors (predicted is wider than gold)

By gold category:

| gold_category | count |
| --- | --- |
| PERSON | 4 |

By predicted category:

| predicted_category | count |
| --- | --- |
| PERSON | 4 |

## Category Confusions (Predicted -> Gold)

| predicted | gold | count |
| --- | --- | --- |
| PERSON | OTHER | 2 |
| OTHER | ADDRESS | 1 |

## FP Examples

### OTHER
- doc `4`: `predicted=OTHER value='Labor Day'`
  - text: `At my suggestion, one morning over breakfast, she agreed, and on the last Sunday before Labor Day we returned to Blaketown by helicopter.`
- doc `33`: `predicted=OTHER value='ΛΑΝΤΖΙΑ'`
  - text: `Frank had given Tyler his address: 03 Πλατεία Μαβίλη 170, ΑΓΛΑΝΤΖΙΑ (ΑΓΛΑΓΓΙΑ)`
- doc `59`: `predicted=OTHER value='Canada'`
  - text: `Taylor Barabás Areavibes Inc 27 2279 President St Gleniti , nan Canada 32586 Mobile: 041-412-293 Desk: 986.291.2294x751 Fax: 001-420-335-7509x38548`
- doc `76`: `predicted=OTHER value='indian'`
  - text: `The restaurant is located at 01 Korte Noordsstraat 307 ÖVERTURINGEN, nan 80815. It serves great indian food.`

### ADDRESS
- doc `22`: `predicted=ADDRESS value='29 March 2017'`
  - text: `On 29 March 2017, the sudanese government formally began the process of withdrawal by invoking Article 50 of the Treaty on European Union`
- doc `59`: `predicted=ADDRESS value='32586'`
  - text: `Taylor Barabás Areavibes Inc 27 2279 President St Gleniti , nan Canada 32586 Mobile: 041-412-293 Desk: 986.291.2294x751 Fax: 001-420-335-7509x38548`

### ORG
- doc `1`: `predicted=ORG value='billing'`
  - text: `billing address: tracy sukhorukova 23 8 wressle road suite 771 polapit tamar nan 77058`
- doc `59`: `predicted=ORG value='Desk'`
  - text: `Taylor Barabás Areavibes Inc 27 2279 President St Gleniti , nan Canada 32586 Mobile: 041-412-293 Desk: 986.291.2294x751 Fax: 001-420-335-7509x38548`

## FN Examples

### OTHER
- doc `47`: `gold=OTHER value='Mrs.'`
  - text: `My name appears incorrectly on credit card statement could you please correct it to Mrs. Amy Kupcová?`
- doc `56`: `gold=OTHER value='Mrs.'`
  - text: `For my take on Mrs. Ayers, see Guilty Pleasures: 5 Musicians Of The 70s You're Supposed To Hate (But Secretly Love)`

## Subset Overlap Examples

### ADDRESS->ADDRESS
- doc `15`: `pred=ADDRESS:'63 30 N. Stadion\nULLINISH' | gold=ADDRESS:'63 30 N. Stadion\nULLINISH\n, nan\n 40839'`
  - text: `I need to add my addresses, here they are: 63 30 N. Stadion ULLINISH , nan 40839, and 75 Auerstrasse 12 Apt. 974 Lyss SÃ£o TomÃ© and PrÃ­ncipe`
- doc `15`: `pred=ADDRESS:'75 Auerstrasse 12 Apt. 974' | gold=ADDRESS:'75 Auerstrasse 12 Apt. 974 Lyss SÃ£o TomÃ© and PrÃ\xadncipe'`
  - text: `I need to add my addresses, here they are: 63 30 N. Stadion ULLINISH , nan 40839, and 75 Auerstrasse 12 Apt. 974 Lyss SÃ£o TomÃ© and PrÃ­ncipe`
- doc `27`: `pred=ADDRESS:'wetegem\n, VOV\n Serbia 23823' | gold=ADDRESS:'52 Ibirapita 8057\nErwetegem\n, VOV\n Serbia 23823'`
  - text: `Please update the billing address with 52 Ibirapita 8057 Erwetegem , VOV Serbia 23823 for this card: 4556156821609393`
- doc `35`: `pred=ADDRESS:'yes Católicos 17\nSaarjärve' | gold=ADDRESS:'06 Reyes Católicos 17\nSaarjärve, JN 91851'`
  - text: `As promised, here's Arne's address: 06 Reyes Católicos 17 Saarjärve, JN 91851`
- doc `49`: `pred=ADDRESS:'Argyll Road\nNoorderwijk\n,' | gold=ADDRESS:'39 79 Argyll Road\nNoorderwijk\n, VAN\n Portugal 76970'`
  - text: `Please return to 39 79 Argyll Road Noorderwijk , VAN Portugal 76970 in case of an issue.`

### OTHER->OTHER
- doc `11`: `pred=OTHER:'http://www.DialForum.co.uk' | gold=OTHER:'http://www.DialForum.co.uk/'`
  - text: `Just posted a photo http://www.DialForum.co.uk/`
- doc `33`: `pred=OTHER:'ΛΑΓΓΙΑ' | gold=OTHER:'ΑΓΛΑΝΤΖΙΑ (ΑΓΛΑΓΓΙΑ)'`
  - text: `Frank had given Tyler his address: 03 Πλατεία Μαβίλη 170, ΑΓΛΑΝΤΖΙΑ (ΑΓΛΑΓΓΙΑ)`
- doc `94`: `pred=OTHER:'Kaplice' | gold=OTHER:'Kaplice 1'`
  - text: `Michael Urner 11 Školní 939 Apt. 827 Kaplice 1 Iceland 80392`

### PHONE->PHONE
- doc `98`: `pred=PHONE:'+1-944-435-4352x047' | gold=PHONE:'+1-944-435-4352x04732'`
  - text: `Tiffany Kilfoyle Nera Economic Consulting 74 106 Kate Edger Place, BINSTED, Vatican City 69 987 88 37 office 001-825-741-0957x5184 fax +1-944-435-4352x04732 mobile`

## Superset Overlap Examples

### PERSON->PERSON
- doc `0`: `pred=PERSON:'ms. Wijtze' | gold=PERSON:'Wijtze'`
  - text: `Please have the manager call me at (96) 627-277 I'd like to join accounts with ms. Wijtze`
- doc `47`: `pred=PERSON:'Mrs. Amy Kupcová' | gold=PERSON:'Amy Kupcová'`
  - text: `My name appears incorrectly on credit card statement could you please correct it to Mrs. Amy Kupcová?`
- doc `56`: `pred=PERSON:'Mrs. Ayers' | gold=PERSON:'Ayers'`
  - text: `For my take on Mrs. Ayers, see Guilty Pleasures: 5 Musicians Of The 70s You're Supposed To Hate (But Secretly Love)`
- doc `85`: `pred=PERSON:'ms. Kuyra' | gold=PERSON:'Kuyra'`
  - text: `Please have the manager call me at 027 896 92 86 I'd like to join accounts with ms. Kuyra`

## Type Mismatch Examples

### PERSON->OTHER
- doc `31`: `pred=PERSON:'Dr.' | gold=OTHER:'Dr.'`
  - text: `Dr. Nielsen is a 50 year old man who grew up in Ústí nad Labem 2.`
- doc `37`: `pred=PERSON:'Mr.' | gold=OTHER:'Mr.'`
  - text: `Mr. Fernandez is a 50 year old man who grew up in Monroe.`

### OTHER->ADDRESS
- doc `1`: `pred=OTHER:'nan' | gold=ADDRESS:'nan'`
  - text: `billing address: tracy sukhorukova 23 8 wressle road suite 771 polapit tamar nan 77058`
