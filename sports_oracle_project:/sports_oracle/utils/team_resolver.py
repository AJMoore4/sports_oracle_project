"""
sports_oracle/utils/team_resolver.py

Canonical team name resolution across all data sources.

PROBLEM:
  BartTorvik says "Duke", ESPN says "Duke Blue Devils",
  CBBD says "Duke Blue Devils", Sports Reference uses "duke".
  Odds API might say "Duke Blue Devils" or "Duke".
  Without a resolver, cross-source joins silently fail.

APPROACH:
  1. Maintain a canonical name for each D1 team
  2. Build an alias index from all known variations
  3. Fuzzy match as last resort (Levenshtein distance)

USAGE:
    resolver = TeamResolver()
    resolver.resolve("Duke Blue Devils")  → "Duke"
    resolver.resolve("DUKE")              → "Duke"
    resolver.resolve("duke")              → "Duke"
    resolver.resolve("UConn")             → "Connecticut"
    resolver.resolve("St. John's (NY)")   → "St. John's"
"""

from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger("sports_oracle.team_resolver")


# ── Canonical team names + known aliases ──────────────────────────────────────
# Canonical name is the KEY. Values are all known aliases across sources.
# This covers every team that's appeared in the NCAA tournament since 2010
# plus all current D1 teams with tricky name variations.
#
# Convention: canonical name = BartTorvik name (short, no mascot)

_TEAM_ALIASES: dict[str, list[str]] = {
    # ── A ──
    "Abilene Christian":  ["Abilene Christian Wildcats", "ACU"],
    "Air Force":          ["Air Force Falcons"],
    "Akron":              ["Akron Zips"],
    "Alabama":            ["Alabama Crimson Tide", "Bama"],
    "Alabama A&M":        ["Alabama A&M Bulldogs", "AAMU"],
    "Alabama St.":        ["Alabama State", "Alabama State Hornets", "Alabama St"],
    "Albany":              ["Albany Great Danes", "Albany (NY)"],
    "Alcorn St.":         ["Alcorn State", "Alcorn State Braves", "Alcorn St"],
    "American":           ["American University", "American Eagles"],
    "Appalachian St.":    ["Appalachian State", "Appalachian State Mountaineers", "App State", "Appalachian St"],
    "Arizona":            ["Arizona Wildcats"],
    "Arizona St.":        ["Arizona State", "Arizona State Sun Devils", "ASU", "Arizona St"],
    "Arkansas":           ["Arkansas Razorbacks", "Ark"],
    "Arkansas St.":       ["Arkansas State", "Arkansas State Red Wolves", "Arkansas St"],
    "Arkansas-Pine Bluff": ["Arkansas-Pine Bluff Golden Lions", "UAPB", "Pine Bluff"],
    "Army":               ["Army Black Knights", "Army West Point"],
    "Auburn":             ["Auburn Tigers"],
    "Austin Peay":        ["Austin Peay Governors", "Austin Peay State"],

    # ── B ──
    "Ball St.":           ["Ball State", "Ball State Cardinals", "Ball St"],
    "Baylor":             ["Baylor Bears"],
    "Bellarmine":         ["Bellarmine Knights"],
    "Belmont":            ["Belmont Bruins"],
    "Bethune-Cookman":    ["Bethune-Cookman Wildcats", "B-CU"],
    "Binghamton":         ["Binghamton Bearcats"],
    "Boise St.":          ["Boise State", "Boise State Broncos", "Boise St"],
    "Boston College":     ["Boston College Eagles", "BC"],
    "Boston University":  ["Boston University Terriers", "BU", "Boston U."],
    "Bowling Green":      ["Bowling Green Falcons", "BGSU", "Bowling Green St."],
    "Bradley":            ["Bradley Braves"],
    "BYU":                ["Brigham Young", "Brigham Young Cougars", "Brigham Young University", "BYU Cougars"],
    "Brown":              ["Brown Bears"],
    "Bryant":             ["Bryant Bulldogs"],
    "Bucknell":           ["Bucknell Bison"],
    "Buffalo":            ["Buffalo Bulls"],
    "Butler":             ["Butler Bulldogs"],

    # ── C ──
    "Cal Baptist":        ["California Baptist", "California Baptist Lancers", "CBU"],
    "Cal Poly":           ["Cal Poly Mustangs", "Cal Poly SLO"],
    "Cal St. Bakersfield":["CSU Bakersfield", "Cal State Bakersfield", "CSUB"],
    "Cal St. Fullerton":  ["CSU Fullerton", "Cal State Fullerton", "CSUF", "Fullerton"],
    "Cal St. Northridge":  ["CSU Northridge", "Cal State Northridge", "CSUN"],
    "California":         ["California Golden Bears", "Cal", "Cal Bears", "UC Berkeley"],
    "Campbell":           ["Campbell Fighting Camels"],
    "Canisius":           ["Canisius Golden Griffins"],
    "Central Arkansas":   ["Central Arkansas Bears", "UCA"],
    "Central Connecticut": ["Central Connecticut State", "Central Connecticut Blue Devils", "CCSU", "Central Conn. St."],
    "Central Michigan":   ["Central Michigan Chippewas", "CMU"],
    "Charleston":         ["College of Charleston", "Charleston Cougars", "Col. of Charleston"],
    "Charleston Southern": ["Charleston Southern Buccaneers"],
    "Charlotte":          ["Charlotte 49ers", "UNC Charlotte"],
    "Chattanooga":        ["Chattanooga Mocs", "UT Chattanooga", "UTC"],
    "Chicago St.":        ["Chicago State", "Chicago State Cougars", "Chicago St"],
    "Cincinnati":         ["Cincinnati Bearcats", "Cincy"],
    "Clemson":            ["Clemson Tigers"],
    "Cleveland St.":      ["Cleveland State", "Cleveland State Vikings", "Cleveland St"],
    "Coastal Carolina":   ["Coastal Carolina Chanticleers", "CCU"],
    "Colgate":            ["Colgate Raiders"],
    "Colorado":           ["Colorado Buffaloes", "CU"],
    "Colorado St.":       ["Colorado State", "Colorado State Rams", "CSU", "Colorado St"],
    "Columbia":           ["Columbia Lions"],
    "Connecticut":        ["Connecticut Huskies", "UConn", "UConn Huskies"],
    "Coppin St.":         ["Coppin State", "Coppin State Eagles", "Coppin St"],
    "Cornell":            ["Cornell Big Red"],
    "Creighton":          ["Creighton Bluejays"],

    # ── D ──
    "Dartmouth":          ["Dartmouth Big Green"],
    "Davidson":           ["Davidson Wildcats"],
    "Dayton":             ["Dayton Flyers"],
    "Delaware":           ["Delaware Fightin' Blue Hens", "Delaware Blue Hens"],
    "Delaware St.":       ["Delaware State", "Delaware State Hornets", "Delaware St"],
    "Denver":             ["Denver Pioneers"],
    "DePaul":             ["DePaul Blue Demons"],
    "Detroit Mercy":      ["Detroit Mercy Titans", "Detroit"],
    "Drake":              ["Drake Bulldogs"],
    "Drexel":             ["Drexel Dragons"],
    "Duke":               ["Duke Blue Devils"],
    "Duquesne":           ["Duquesne Dukes"],

    # ── E ──
    "East Carolina":      ["East Carolina Pirates", "ECU"],
    "East Tennessee St.": ["East Tennessee State", "ETSU", "East Tennessee State Buccaneers", "East Tennessee St"],
    "Eastern Illinois":   ["Eastern Illinois Panthers", "EIU"],
    "Eastern Kentucky":   ["Eastern Kentucky Colonels", "EKU"],
    "Eastern Michigan":   ["Eastern Michigan Eagles", "EMU"],
    "Eastern Washington": ["Eastern Washington Eagles", "EWU"],
    "Elon":               ["Elon Phoenix"],
    "Evansville":         ["Evansville Purple Aces"],

    # ── F ──
    "Fairfield":          ["Fairfield Stags"],
    "Fairleigh Dickinson": ["Fairleigh Dickinson Knights", "FDU"],
    "FGCU":               ["Florida Gulf Coast", "Florida Gulf Coast Eagles"],
    "Florida":            ["Florida Gators", "UF"],
    "Florida A&M":        ["Florida A&M Rattlers", "FAMU"],
    "Florida Atlantic":   ["Florida Atlantic Owls", "FAU"],
    "Florida International": ["FIU", "FIU Panthers", "Florida Intl."],
    "Florida St.":        ["Florida State", "Florida State Seminoles", "FSU", "Florida St"],
    "Fordham":            ["Fordham Rams"],
    "Fresno St.":         ["Fresno State", "Fresno State Bulldogs", "Fresno St"],
    "Furman":             ["Furman Paladins"],

    # ── G ──
    "Gardner-Webb":       ["Gardner-Webb Runnin' Bulldogs", "Gardner Webb"],
    "George Mason":       ["George Mason Patriots", "GMU"],
    "George Washington":  ["George Washington Colonials", "GW", "GWU", "GW Revolutionaries", "George Washington Revolutionaries"],
    "Georgetown":         ["Georgetown Hoyas"],
    "Georgia":            ["Georgia Bulldogs", "UGA"],
    "Georgia Southern":   ["Georgia Southern Eagles"],
    "Georgia St.":        ["Georgia State", "Georgia State Panthers", "Georgia St"],
    "Georgia Tech":       ["Georgia Tech Yellow Jackets", "GT"],
    "Gonzaga":            ["Gonzaga Bulldogs", "Zags"],
    "Grambling":          ["Grambling State", "Grambling State Tigers", "Grambling St."],
    "Grand Canyon":       ["Grand Canyon Antelopes", "GCU"],
    "Green Bay":          ["Green Bay Phoenix", "UW-Green Bay", "Wisconsin-Green Bay"],

    # ── H ──
    "Hampton":            ["Hampton Pirates"],
    "Hartford":           ["Hartford Hawks"],
    "Harvard":            ["Harvard Crimson"],
    "Hawaii":             ["Hawai'i", "Hawaii Rainbow Warriors", "Hawai'i Rainbow Warriors"],
    "High Point":         ["High Point Panthers"],
    "Hofstra":            ["Hofstra Pride"],
    "Holy Cross":         ["Holy Cross Crusaders"],
    "Houston":            ["Houston Cougars", "UH"],
    "Houston Christian":  ["Houston Christian Huskies", "Houston Baptist", "HCU"],
    "Howard":             ["Howard Bison"],

    # ── I ──
    "Idaho":              ["Idaho Vandals"],
    "Idaho St.":          ["Idaho State", "Idaho State Bengals", "Idaho St"],
    "Illinois":           ["Illinois Fighting Illini"],
    "Illinois St.":       ["Illinois State", "Illinois State Redbirds", "Illinois St"],
    "Incarnate Word":     ["UIW", "Incarnate Word Cardinals"],
    "Indiana":            ["Indiana Hoosiers", "IU"],
    "Indiana St.":        ["Indiana State", "Indiana State Sycamores", "Indiana St"],
    "Iona":               ["Iona Gaels"],
    "Iowa":               ["Iowa Hawkeyes"],
    "Iowa St.":           ["Iowa State", "Iowa State Cyclones", "Iowa St", "ISU"],
    "IUPUI":              ["IUPUI Jaguars"],

    # ── J ──
    "Jackson St.":        ["Jackson State", "Jackson State Tigers", "Jackson St"],
    "Jacksonville":       ["Jacksonville Dolphins", "JU"],
    "Jacksonville St.":   ["Jacksonville State", "Jacksonville State Gamecocks", "Jacksonville St", "Jax State"],
    "James Madison":      ["James Madison Dukes", "JMU"],

    # ── K ──
    "Kansas":             ["Kansas Jayhawks", "KU"],
    "Kansas St.":         ["Kansas State", "Kansas State Wildcats", "K-State", "Kansas St", "KSU"],
    "Kennesaw St.":       ["Kennesaw State", "Kennesaw State Owls", "Kennesaw St"],
    "Kent St.":           ["Kent State", "Kent State Golden Flashes", "Kent St"],
    "Kentucky":           ["Kentucky Wildcats", "UK"],

    # ── L ──
    "La Salle":           ["La Salle Explorers"],
    "Lafayette":          ["Lafayette Leopards"],
    "Lamar":              ["Lamar Cardinals"],
    "Lehigh":             ["Lehigh Mountain Hawks"],
    "Liberty":            ["Liberty Flames"],
    "Lindenwood":         ["Lindenwood Lions"],
    "Lipscomb":           ["Lipscomb Bisons"],
    "Little Rock":        ["Little Rock Trojans", "UALR", "Arkansas-Little Rock"],
    "Long Beach St.":     ["Long Beach State", "Long Beach State 49ers", "Long Beach St", "LBSU"],
    "Long Island":        ["LIU", "Long Island University", "LIU Sharks"],
    "Longwood":           ["Longwood Lancers"],
    "Louisiana":          ["Louisiana Ragin' Cajuns", "UL Lafayette", "Louisiana-Lafayette"],
    "Louisiana Tech":     ["Louisiana Tech Bulldogs", "LA Tech"],
    "Louisville":         ["Louisville Cardinals", "UofL"],
    "Loyola (MD)":        ["Loyola Maryland", "Loyola Maryland Greyhounds"],
    "Loyola Chicago":     ["Loyola (IL)", "Loyola-Chicago", "Loyola Ramblers", "Loyola (Chi)"],
    "Loyola Marymount":   ["Loyola Marymount Lions", "LMU"],
    "LSU":                ["Louisiana State", "LSU Tigers", "Louisiana State Tigers"],

    # ── M ──
    "Maine":              ["Maine Black Bears"],
    "Manhattan":          ["Manhattan Jaspers"],
    "Marist":             ["Marist Red Foxes"],
    "Marquette":          ["Marquette Golden Eagles"],
    "Marshall":           ["Marshall Thundering Herd"],
    "Maryland":           ["Maryland Terrapins", "Terps", "UMD"],
    "Maryland-Eastern Shore": ["Maryland-Eastern Shore Hawks", "UMES", "MD-Eastern Shore"],
    "Massachusetts":      ["UMass", "UMass Minutemen", "Mass", "Massachusetts Minutemen"],
    "McNeese":            ["McNeese State", "McNeese Cowboys", "McNeese St."],
    "Memphis":            ["Memphis Tigers"],
    "Mercer":             ["Mercer Bears"],
    "Merrimack":          ["Merrimack Warriors"],
    "Miami (FL)":         ["Miami", "Miami Hurricanes", "Miami (Fla.)"],
    "Miami (OH)":         ["Miami Ohio", "Miami (Ohio)", "Miami OH RedHawks", "Miami RedHawks"],
    "Michigan":           ["Michigan Wolverines"],
    "Michigan St.":       ["Michigan State", "Michigan State Spartans", "MSU", "Michigan St"],
    "Middle Tennessee":   ["Middle Tennessee Blue Raiders", "MTSU", "Middle Tennessee St."],
    "Milwaukee":          ["UW-Milwaukee", "Milwaukee Panthers", "Wisconsin-Milwaukee"],
    "Minnesota":          ["Minnesota Golden Gophers"],
    "Mississippi St.":    ["Mississippi State", "Mississippi State Bulldogs", "Miss State", "Mississippi St"],
    "Mississippi Valley St.": ["Mississippi Valley State", "MVSU", "Mississippi Valley St"],
    "Missouri":           ["Missouri Tigers", "Mizzou"],
    "Missouri St.":       ["Missouri State", "Missouri State Bears", "Missouri St"],
    "Monmouth":           ["Monmouth Hawks"],
    "Montana":            ["Montana Grizzlies"],
    "Montana St.":        ["Montana State", "Montana State Bobcats", "Montana St"],
    "Morehead St.":       ["Morehead State", "Morehead State Eagles", "Morehead St"],
    "Morgan St.":         ["Morgan State", "Morgan State Bears", "Morgan St"],
    "Mount St. Mary's":   ["Mount St. Mary's Mountaineers", "Mt. St. Mary's", "Mount St Mary's"],
    "Murray St.":         ["Murray State", "Murray State Racers", "Murray St"],

    # ── N ──
    "Navy":               ["Navy Midshipmen"],
    "NC State":           ["North Carolina State", "NC State Wolfpack", "N.C. State"],
    "Nebraska":           ["Nebraska Cornhuskers"],
    "Nevada":             ["Nevada Wolf Pack"],
    "New Hampshire":      ["New Hampshire Wildcats", "UNH"],
    "New Mexico":         ["New Mexico Lobos", "UNM"],
    "New Mexico St.":     ["New Mexico State", "New Mexico State Aggies", "NMSU", "New Mexico St"],
    "New Orleans":        ["New Orleans Privateers", "UNO"],
    "Niagara":            ["Niagara Purple Eagles"],
    "Nicholls St.":       ["Nicholls State", "Nicholls Colonels", "Nicholls", "Nicholls St"],
    "NJIT":               ["NJIT Highlanders", "New Jersey Tech"],
    "Norfolk St.":        ["Norfolk State", "Norfolk State Spartans", "Norfolk St"],
    "North Alabama":      ["North Alabama Lions", "UNA"],
    "North Carolina":     ["North Carolina Tar Heels", "UNC", "Carolina"],
    "North Carolina A&T": ["NC A&T", "North Carolina A&T Aggies"],
    "North Carolina Central": ["NC Central", "NCCU", "North Carolina Central Eagles"],
    "North Dakota":       ["North Dakota Fighting Hawks", "UND"],
    "North Dakota St.":   ["North Dakota State", "NDSU", "North Dakota State Bison", "North Dakota St"],
    "North Florida":      ["North Florida Ospreys", "UNF"],
    "North Texas":        ["North Texas Mean Green", "UNT"],
    "Northeastern":       ["Northeastern Huskies"],
    "Northern Arizona":   ["Northern Arizona Lumberjacks", "NAU"],
    "Northern Colorado":  ["Northern Colorado Bears", "UNC Bears"],
    "Northern Illinois":  ["Northern Illinois Huskies", "NIU"],
    "Northern Iowa":      ["Northern Iowa Panthers", "UNI"],
    "Northern Kentucky":  ["Northern Kentucky Norse", "NKU"],
    "Northwestern":       ["Northwestern Wildcats"],
    "Northwestern St.":   ["Northwestern State", "Northwestern State Demons", "Northwestern St"],
    "Notre Dame":         ["Notre Dame Fighting Irish"],

    # ── O ──
    "Oakland":            ["Oakland Golden Grizzlies"],
    "Ohio":               ["Ohio Bobcats", "Ohio University"],
    "Ohio St.":           ["Ohio State", "Ohio State Buckeyes", "OSU", "Ohio St"],
    "Oklahoma":           ["Oklahoma Sooners", "OU"],
    "Oklahoma St.":       ["Oklahoma State", "Oklahoma State Cowboys", "Oklahoma St"],
    "Old Dominion":       ["Old Dominion Monarchs", "ODU"],
    "Ole Miss":           ["Mississippi", "Mississippi Rebels"],
    "Omaha":              ["Nebraska-Omaha", "Omaha Mavericks", "UNO Mavericks"],
    "Oral Roberts":       ["Oral Roberts Golden Eagles", "ORU"],
    "Oregon":             ["Oregon Ducks"],
    "Oregon St.":         ["Oregon State", "Oregon State Beavers", "Oregon St"],

    # ── P ──
    "Pacific":            ["Pacific Tigers"],
    "Penn":               ["Pennsylvania", "Penn Quakers"],
    "Penn St.":           ["Penn State", "Penn State Nittany Lions", "Penn St", "PSU"],
    "Pepperdine":         ["Pepperdine Waves"],
    "Pittsburgh":         ["Pittsburgh Panthers", "Pitt"],
    "Portland":           ["Portland Pilots"],
    "Portland St.":       ["Portland State", "Portland State Vikings", "Portland St"],
    "Prairie View A&M":   ["Prairie View A&M Panthers", "Prairie View"],
    "Presbyterian":       ["Presbyterian Blue Hose"],
    "Princeton":          ["Princeton Tigers"],
    "Providence":         ["Providence Friars"],
    "Purdue":             ["Purdue Boilermakers"],

    # ── Q ──
    "Queens":             ["Queens Royals", "Queens University"],
    "Quinnipiac":         ["Quinnipiac Bobcats"],

    # ── R ──
    "Radford":            ["Radford Highlanders"],
    "Rhode Island":       ["Rhode Island Rams", "URI"],
    "Rice":               ["Rice Owls"],
    "Richmond":           ["Richmond Spiders"],
    "Rider":              ["Rider Broncs"],
    "Robert Morris":      ["Robert Morris Colonials", "RMU"],
    "Rutgers":            ["Rutgers Scarlet Knights"],

    # ── S ──
    "Sacramento St.":     ["Sacramento State", "Sacramento State Hornets", "Sac State", "Sacramento St"],
    "Sacred Heart":       ["Sacred Heart Pioneers", "SHU"],
    "Saint Joseph's":     ["Saint Joseph's Hawks", "St. Joseph's", "St Joseph's"],
    "Saint Louis":        ["Saint Louis Billikens", "SLU", "St. Louis"],
    "Saint Mary's":       ["Saint Mary's Gaels", "St. Mary's", "SMC", "Saint Mary's (CA)"],
    "Saint Peter's":      ["Saint Peter's Peacocks", "St. Peter's"],
    "Sam Houston St.":    ["Sam Houston", "Sam Houston State", "Sam Houston Bearkats", "SHSU", "Sam Houston St"],
    "Samford":            ["Samford Bulldogs"],
    "San Diego":          ["San Diego Toreros"],
    "San Diego St.":      ["San Diego State", "San Diego State Aztecs", "SDSU", "San Diego St"],
    "San Francisco":      ["San Francisco Dons", "USF"],
    "San Jose St.":       ["San Jose State", "San Jose State Spartans", "SJSU", "San Jose St", "San José St", "San José St Spartans", "San José State"],
    "Santa Clara":        ["Santa Clara Broncos"],
    "Seattle":            ["Seattle Redhawks", "Seattle University", "Seattle U"],
    "Seton Hall":         ["Seton Hall Pirates"],
    "Siena":              ["Siena Saints"],
    "South Alabama":      ["South Alabama Jaguars", "USA"],
    "South Carolina":     ["South Carolina Gamecocks"],
    "South Carolina St.": ["South Carolina State", "SC State", "South Carolina St"],
    "South Dakota":       ["South Dakota Coyotes", "USD"],
    "South Dakota St.":   ["South Dakota State", "SDSU Jackrabbits", "South Dakota St"],
    "South Florida":      ["South Florida Bulls", "USF Bulls"],
    "Southeast Missouri St.": ["Southeast Missouri State", "SEMO", "Southeast Missouri St"],
    "Southeastern Louisiana": ["Southeastern Louisiana Lions", "SE Louisiana"],
    "Southern":           ["Southern University", "Southern Jaguars", "Southern U."],
    "Southern Illinois":  ["Southern Illinois Salukis", "SIU"],
    "Southern Indiana":   ["Southern Indiana Screaming Eagles", "USI"],
    "Southern Methodist": ["SMU", "SMU Mustangs"],
    "Southern Miss":      ["Southern Mississippi", "Southern Miss Golden Eagles", "USM"],
    "Southern Utah":      ["Southern Utah Thunderbirds", "SUU"],
    "St. Bonaventure":    ["St. Bonaventure Bonnies", "Saint Bonaventure"],
    "St. Francis (PA)":   ["Saint Francis", "Saint Francis Red Flash"],
    "St. John's":         ["St. John's Red Storm", "Saint John's", "St. John's (NY)"],
    "St. Thomas":         ["St. Thomas Tommies", "St. Thomas (MN)"],
    "Stanford":           ["Stanford Cardinal"],
    "Stephen F. Austin":  ["Stephen F. Austin Lumberjacks", "SFA"],
    "Stetson":            ["Stetson Hatters"],
    "Stonehill":          ["Stonehill Skyhawks"],
    "Stony Brook":        ["Stony Brook Seawolves"],
    "Syracuse":           ["Syracuse Orange", "Cuse"],

    # ── T ──
    "Tarleton St.":       ["Tarleton State", "Tarleton State Texans", "Tarleton St"],
    "TCU":                ["Texas Christian", "TCU Horned Frogs"],
    "Temple":             ["Temple Owls"],
    "Tennessee":          ["Tennessee Volunteers", "Vols"],
    "Tennessee St.":      ["Tennessee State", "Tennessee State Tigers", "Tennessee St"],
    "Tennessee Tech":     ["Tennessee Tech Golden Eagles", "TTU"],
    "Texas":              ["Texas Longhorns", "UT"],
    "Texas A&M":          ["Texas A&M Aggies", "TAMU"],
    "Texas A&M-CC":       ["Texas A&M-Corpus Christi", "Texas A&M-Corpus Christi Islanders", "A&M-Corpus Christi"],
    "Texas Southern":     ["Texas Southern Tigers", "TSU"],
    "Texas St.":          ["Texas State", "Texas State Bobcats", "Texas St"],
    "Texas Tech":         ["Texas Tech Red Raiders", "TTU Red Raiders"],
    "The Citadel":        ["Citadel Bulldogs", "Citadel"],
    "Toledo":             ["Toledo Rockets"],
    "Towson":             ["Towson Tigers"],
    "Troy":               ["Troy Trojans"],
    "Tulane":             ["Tulane Green Wave"],
    "Tulsa":              ["Tulsa Golden Hurricane"],

    # ── U ──
    "UAB":                ["UAB Blazers", "Alabama-Birmingham"],
    "UC Davis":           ["UC Davis Aggies"],
    "UC Irvine":          ["UC Irvine Anteaters", "UCI"],
    "UC Riverside":       ["UC Riverside Highlanders", "UCR"],
    "UC San Diego":       ["UC San Diego Tritons", "UCSD"],
    "UC Santa Barbara":   ["UC Santa Barbara Gauchos", "UCSB"],
    "UCF":                ["Central Florida", "UCF Knights"],
    "UCLA":               ["UCLA Bruins"],
    "UIC":                ["UIC Flames", "Illinois-Chicago"],
    "UMass Lowell":       ["UMass Lowell River Hawks"],
    "UMBC":               ["UMBC Retrievers", "Maryland-Baltimore County"],
    "UMKC":               ["Kansas City", "Kansas City Roos", "Missouri-Kansas City"],
    "UNC Asheville":      ["UNC Asheville Bulldogs", "UNC-Asheville"],
    "UNC Greensboro":     ["UNC Greensboro Spartans", "UNCG"],
    "UNC Wilmington":     ["UNC Wilmington Seahawks", "UNCW"],
    "UNLV":               ["UNLV Rebels", "Nevada-Las Vegas"],
    "USC":                ["Southern California", "USC Trojans"],
    "USC Upstate":        ["USC Upstate Spartans"],
    "UT Arlington":       ["UT Arlington Mavericks", "Texas-Arlington", "UTA", "UT-Arlington", "UT-Arlington Mavericks"],
    "UT Martin":          ["UT Martin Skyhawks", "Tennessee-Martin", "UTM"],
    "UT Rio Grande Valley": ["UTRGV", "UTRGV Vaqueros", "Texas-Rio Grande Valley"],
    "Utah":               ["Utah Utes"],
    "Utah St.":           ["Utah State", "Utah State Aggies", "Utah St", "USU"],
    "Utah Tech":          ["Utah Tech Trailblazers", "Dixie State"],
    "Utah Valley":        ["Utah Valley Wolverines", "UVU"],
    "UTEP":               ["UTEP Miners", "Texas-El Paso"],
    "UTSA":               ["UTSA Roadrunners", "Texas-San Antonio"],

    # ── V ──
    "Valparaiso":         ["Valparaiso Beacons", "Valpo"],
    "Vanderbilt":         ["Vanderbilt Commodores", "Vandy"],
    "VCU":                ["VCU Rams", "Virginia Commonwealth"],
    "Vermont":            ["Vermont Catamounts"],
    "Villanova":          ["Villanova Wildcats", "Nova"],
    "Virginia":           ["Virginia Cavaliers", "UVA"],
    "Virginia Tech":      ["Virginia Tech Hokies", "VT"],
    "VMI":                ["VMI Keydets", "Virginia Military"],

    # ── W ──
    "Wagner":             ["Wagner Seahawks"],
    "Wake Forest":        ["Wake Forest Demon Deacons", "Wake"],
    "Washington":         ["Washington Huskies", "UW"],
    "Washington St.":     ["Washington State", "Washington State Cougars", "Wazzu", "Washington St", "WSU"],
    "Weber St.":          ["Weber State", "Weber State Wildcats", "Weber St"],
    "West Virginia":      ["West Virginia Mountaineers", "WVU"],
    "Western Carolina":   ["Western Carolina Catamounts", "WCU"],
    "Western Illinois":   ["Western Illinois Leathernecks", "WIU"],
    "Western Kentucky":   ["Western Kentucky Hilltoppers", "WKU"],
    "Western Michigan":   ["Western Michigan Broncos", "WMU"],
    "Wichita St.":        ["Wichita State", "Wichita State Shockers", "Wichita St"],
    "William & Mary":     ["William & Mary Tribe", "W&M"],
    "Winthrop":           ["Winthrop Eagles"],
    "Wisconsin":          ["Wisconsin Badgers"],
    "Wofford":            ["Wofford Terriers"],
    "Wright St.":         ["Wright State", "Wright State Raiders", "Wright St"],
    "Wyoming":            ["Wyoming Cowboys"],

    # ── X/Y ──
    "Xavier":             ["Xavier Musketeers"],
    "Yale":               ["Yale Bulldogs"],
    "Youngstown St.":     ["Youngstown State", "Youngstown State Penguins", "Youngstown St"],
}


class TeamResolver:
    """
    Resolves team name variations to canonical names.
    Supports exact match, alias lookup, and fuzzy matching.
    """

    def __init__(self):
        # Build reverse index: alias → canonical
        self._alias_index: dict[str, str] = {}
        self._canonical_set: set[str] = set()

        for canonical, aliases in _TEAM_ALIASES.items():
            canon_key = canonical.lower().strip()
            self._alias_index[canon_key] = canonical
            self._canonical_set.add(canonical)

            for alias in aliases:
                alias_key = alias.lower().strip()
                self._alias_index[alias_key] = canonical

        logger.info(
            f"TeamResolver initialized: {len(self._canonical_set)} teams, "
            f"{len(self._alias_index)} aliases indexed"
        )

    def resolve(self, name: str) -> Optional[str]:
        """
        Resolve a team name to its canonical form.
        Returns None if no match found.

        Tries in order:
          1. Exact match (case-insensitive)
          2. Normalized match (strip punctuation, abbreviations)
          3. Fuzzy match (edit distance ≤ 3)
        """
        if not name:
            return None

        key = name.lower().strip()

        # 1. Exact match
        if key in self._alias_index:
            return self._alias_index[key]

        # 2. Normalized match — strip common suffixes and punctuation
        normalized = self._normalize(key)
        if normalized in self._alias_index:
            return self._alias_index[normalized]

        # 3. Try without "state", "st.", etc.
        for variant in self._generate_variants(key):
            if variant in self._alias_index:
                return self._alias_index[variant]

        # 4. Fuzzy match — edit distance
        best_match = self._fuzzy_match(key, max_distance=3)
        if best_match:
            logger.debug(f"Fuzzy matched '{name}' → '{best_match}'")
            return best_match

        logger.warning(f"Could not resolve team name: '{name}'")
        return None

    def resolve_or_original(self, name: str) -> str:
        """Resolve, or return the original name if no match."""
        return self.resolve(name) or name

    def is_known(self, name: str) -> bool:
        """Check if a team name can be resolved."""
        return self.resolve(name) is not None

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """Strip punctuation, extra whitespace, common suffixes."""
        # Remove parenthetical qualifiers: "(NY)", "(FL)", etc.
        name = re.sub(r"\s*\([^)]*\)\s*", "", name)
        # Remove punctuation except hyphens and ampersands
        name = re.sub(r"[.'']", "", name)
        # Normalize whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name.lower()

    @staticmethod
    def _generate_variants(key: str) -> list[str]:
        """Generate common name variants to try."""
        variants = []

        # "State" ↔ "St."
        if " state" in key:
            variants.append(key.replace(" state", " st."))
            variants.append(key.replace(" state", " st"))
        if " st." in key:
            variants.append(key.replace(" st.", " state"))
            variants.append(key.replace(" st.", " st"))
        if " st" in key and " st." not in key and " state" not in key:
            variants.append(key.replace(" st", " state"))
            variants.append(key.replace(" st", " st."))

        # Strip mascot (last word if >2 words)
        words = key.split()
        if len(words) > 2:
            variants.append(" ".join(words[:-1]))

        # "University of X" → "X"
        if key.startswith("university of "):
            variants.append(key.replace("university of ", ""))

        # Strip "the "
        if key.startswith("the "):
            variants.append(key[4:])

        return variants

    def _fuzzy_match(self, key: str, max_distance: int = 3) -> Optional[str]:
        """
        Find closest match using Levenshtein distance.
        Only checks canonical names (not all aliases) for speed.
        """
        best = None
        best_dist = max_distance + 1

        for canonical in self._canonical_set:
            dist = self._edit_distance(key, canonical.lower(), max_distance)
            if dist < best_dist:
                best_dist = dist
                best = canonical

        return best if best_dist <= max_distance else None

    @staticmethod
    def _edit_distance(s1: str, s2: str, max_dist: int = 5) -> int:
        """
        Levenshtein distance with early termination.
        Returns max_dist+1 if actual distance exceeds max_dist.
        """
        if abs(len(s1) - len(s2)) > max_dist:
            return max_dist + 1

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        prev = list(range(len(s1) + 1))
        for j in range(1, len(s2) + 1):
            curr = [j] + [0] * len(s1)
            for i in range(1, len(s1) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr[i] = min(
                    curr[i - 1] + 1,
                    prev[i] + 1,
                    prev[i - 1] + cost,
                )
            if min(curr) > max_dist:
                return max_dist + 1
            prev = curr

        return prev[len(s1)]

    # ── Bulk operations ───────────────────────────────────────────────────

    def resolve_list(self, names: list[str]) -> list[Optional[str]]:
        """Resolve a list of team names."""
        return [self.resolve(n) for n in names]

    def get_canonical_name(self, name: str) -> str:
        """Alias for resolve_or_original."""
        return self.resolve_or_original(name)

    @property
    def all_teams(self) -> list[str]:
        """Return sorted list of all canonical team names."""
        return sorted(self._canonical_set)

    @property
    def team_count(self) -> int:
        return len(self._canonical_set)


# ── Module-level singleton ────────────────────────────────────────────────────
_resolver: Optional[TeamResolver] = None


def get_resolver() -> TeamResolver:
    """Get or create the module-level TeamResolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = TeamResolver()
    return _resolver


def resolve_team(name: str) -> Optional[str]:
    """Convenience function — resolve a team name using the singleton."""
    return get_resolver().resolve(name)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    r = TeamResolver()

    test_cases = [
        "Duke",
        "Duke Blue Devils",
        "DUKE",
        "duke",
        "UConn",
        "Connecticut Huskies",
        "St. John's (NY)",
        "St. John's",
        "Michigan State Spartans",
        "Michigan St.",
        "Michigan St",
        "Gonzaga Bulldogs",
        "Zags",
        "North Carolina Tar Heels",
        "UNC",
        "Carolina",
        "Florida Atlantic Owls",
        "FAU",
        "TCU Horned Frogs",
        "Texas Christian",
        "totally fake team",
    ]

    print(f"\n🏀 TeamResolver — {r.team_count} teams, "
          f"{len(r._alias_index)} aliases")
    print("=" * 50)
    for name in test_cases:
        result = r.resolve(name)
        status = "✅" if result else "❌"
        print(f"  {status} '{name}' → {result}")
