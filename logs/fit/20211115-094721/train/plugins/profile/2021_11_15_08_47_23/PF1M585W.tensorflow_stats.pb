"?
BHostIDLE"IDLE1????E?AA????E?AaX????iX?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????L?@9????L?@A????L?@I????L?@a?7?????i??H)Β???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??????y@9?????3@A     ?w@I?0?02@a9r=a?yV?iO:?????Unknown
dHostDataset"Iterator::Model(1??????@9uPuP:@A?????$v@I?_??0@a R?:?U?ix??|?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1fffff?s@9??	??	.@Afffff?s@I??	??	.@a?E?R?i?a??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??????p@9????)@A??????p@I????)@a{??ŹP?i?9?{?????Unknown
vHost_FusedMatMul"sequential_1/dense_1/BiasAdd(1?????In@98?8?'@A?????In@I8?8?'@a????L?i- ?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????	h@9uPuP"@A?????	h@IuPuP"@a6q}???F?i??p?????Unknown
^	HostGatherV2"GatherV2(1fffff?d@9	??	??@Afffff?d@I	??	??@a2?X???C?i??X??????Unknown
`
HostGatherV2"
GatherV2_1(1?????Ib@9??????@A?????Ib@I??????@a??,!ZA?i?)?[????Unknown
gHostStridedSlice"strided_slice(1fffff?`@9?؏?؏@Afffff?`@I?؏?؏@a??x???i-?
????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1?????\`@9??????@A?????\`@I??????@a/?????ii??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?@9?0?0LB@A??????^@I??????@aA??e?c=?i?,?????Unknown
lHostIteratorGetNext"IteratorGetNext(1?????l]@9?Fk?Fk@A?????l]@I?Fk?Fk@a????|?;?i?;S?!????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1??????[@9uPuP@A??????[@IuPuP@a??D/0:?i??;?g????Unknown
[HostAddV2"Adam/add(1     `[@9?m۶m?@A     `[@I?m۶m?@a?V???9?iu?9Ӧ????Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1??????Z@9˥\ʥ\@A??????Z@I˥\ʥ\@aH ?~?[9?i?X?G?????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(1????̌X@9Gk?Fk?@A????̌X@IGk?Fk?@a'Kd?RK7?ib?$??????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1?????9X@9QuPu@A?????9X@IQuPu@a=?$a?6?i5HI>?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice(1     ?V@9I?$I?$@A     ?V@II?$I?$@aa??/^Y5?iQAjF????Unknown
?HostSquaredDifference"$mean_squared_error/SquaredDifference(1      T@9??y??y@A      T@I??y??y@aVX$??2?i??i??????Unknown
?HostCast"Msequential_1/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1fffff?S@9	??	??@Afffff?S@I	??	??@a??(???2?i?????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate(1     ?d@9n۶m۶@A      S@II?$I?$@a?Z?%2?i???6;????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1fffff&R@9?:??:?@Afffff&R@I?:??:?@a 9??81?i???Mb????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     m@9I?$I?$&@A     ?P@II?$I?$	@a??Hy?O/?iwC?KW????Unknown
}HostMaximum"(gradient_tape/mean_squared_error/Maximum(1??????K@9?8?8@A??????K@I?8?8@as?Q4?l*?i????????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(*1????̌@@9?8?8??A????̌@@I?8?8??a=??)h?i&Q?[?????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@aD$q??iH?\O?????Unknown?
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a=?HH???>i?o??????Unknown?
aHostIdentity"Identity(1????????9????????A????????I????????a??H?!??>i?????????Unknown?*?
uHostFlushSummaryWriter"FlushSummaryWriter(1????L?@9????L?@A????L?@I????L?@a.S9T????i.S9T?????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??????y@9?????3@A     ?w@I?0?02@aP?Ӕ???it???????Unknown
dHostDataset"Iterator::Model(1??????@9uPuP:@A?????$v@I?_??0@aT)?H????i[u??F???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1fffff?s@9??	??	.@Afffff?s@I??	??	.@a??Y?????i?CZ?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??????p@9????)@A??????p@I????)@apJYgՐ?i\???Gj???Unknown
vHost_FusedMatMul"sequential_1/dense_1/BiasAdd(1?????In@98?8?'@A?????In@I8?8?'@a?4?2??i?i?_?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????	h@9uPuP"@A?????	h@IuPuP"@a??bR????i??(?*B???Unknown
^HostGatherV2"GatherV2(1fffff?d@9	??	??@Afffff?d@I	??	??@a???\????i?ԛ??????Unknown
`	HostGatherV2"
GatherV2_1(1?????Ib@9??????@A?????Ib@I??????@a????*+??ig?/V?????Unknown
g
HostStridedSlice"strided_slice(1fffff?`@9?؏?؏@Afffff?`@I?؏?؏@a???	|???i?VFP ???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1?????\`@9??????@A?????\`@I??????@a??*+?A??i???Va???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?@9?0?0LB@A??????^@I??????@a?R???~?iT????????Unknown
lHostIteratorGetNext"IteratorGetNext(1?????l]@9?Fk?Fk@A?????l]@I?Fk?Fk@a???`?;}?ii?}UZ????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1??????[@9uPuP@A??????[@IuPuP@a"⦫k{?i?^ˬ1???Unknown
[HostAddV2"Adam/add(1     `[@9?m۶m?@A     `[@I?m۶m?@a???q2{?i?????F???Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1??????Z@9˥\ʥ\@A??????Z@I˥\ʥ\@a?V?i ?z?iVnѰ{???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(1????̌X@9Gk?Fk?@A????̌X@IGk?Fk?@a?,?0?cx?i?wϵx????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1?????9X@9QuPu@A?????9X@IQuPu@aN??qIx?iS?H?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice(1     ?V@9I?$I?$@A     ?V@II?$I?$@a,?Y??Zv?i??!fP	???Unknown
?HostSquaredDifference"$mean_squared_error/SquaredDifference(1      T@9??y??y@A      T@I??y??y@a?*l???s?i	???1???Unknown
?HostCast"Msequential_1/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1fffff?S@9	??	??@Afffff?S@I	??	??@a\????es?i?|???W???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate(1     ?d@9n۶m۶@A      S@II?$I?$@aXo??, s?i?????}???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1fffff&R@9?:??:?@Afffff&R@I?:??:?@a?\eE2r?i:?1d?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     m@9I?$I?$&@A     ?P@II?$I?$	@a?<?ʊdp?i?8?y?????Unknown
}HostMaximum"(gradient_tape/mean_squared_error/Maximum(1??????K@9?8?8@A??????K@I?8?8@aDl&A?k?i L??^????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(*1????̌@@9?8?8??A????̌@@I?8?8??a???JBq`?i	?7??????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@a?Ҩ?Z?iE?9????Unknown?
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a????9?i?Ɓt????Unknown?
aHostIdentity"Identity(1????????9????????A????????I????????a??+?O|?i      ???Unknown?