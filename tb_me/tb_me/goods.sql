CREATE TABLE `goods` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) DEFAULT NULL,
  `commout` varchar(255) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL,
  `confirm_goods_count` varchar(255) DEFAULT NULL,
  `send_city` varchar(255) DEFAULT NULL,
  `info` varchar(2550) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `itemid` varchar(255) DEFAULT NULL,
  `detail_count` varchar(255) DEFAULT NULL,
  `href` varchar(255) DEFAULT NULL,
  `tb_price` decimal(10,2) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8