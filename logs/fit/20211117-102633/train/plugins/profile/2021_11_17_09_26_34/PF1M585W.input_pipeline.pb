$	????S?????ه???!?rh????!l	??g???$	???"?@gؠ?q@nZ??ޔ @!? ??_?.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?S㥛????JY?8ֵ?A?sF????Y?;Nё\??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5?;N????Q???A/?$???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`??"????#??~j???A?;Nё\??Y??ͪ?զ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n??EGr????A?/?'??YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1??%?????~j?t??A?]K?=??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o?????Y??ڊ??A?Zd;???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'1?Z???X?? ??A;pΈ????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-C??6???-?????A?Pk?w???Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO??g??j+???A?&1???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	 ?~?:p???????B??Ab??4?8??Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
Y?? ???????H??Ay?&1???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A?c?]K?????S???Ao???T???Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;????镲q??Az6?>W[??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L??$???????AD????9??Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]???? ???A??(????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????X????Aj?t???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ϊ??V???x$(~??A??ZӼ???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O????^K?=???A??1??%??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&l	??g?????H.???AO??e?c??Y???1段?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????B???=
ףp=??A??e?c]??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?rh??????W?2ı?AjM??St??YP?s???*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatΪ??V???!???8?@)sh??|???1?~???<@:Preprocessing2F
Iterator::Model??7??d??!;	??N?A@)?c]?F??1?zO?7|7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?z?G???!d?E[J*@)?z?G???1d?E[J*@:Preprocessing2U
Iterator::Model::ParallelMapV2x$(~???!?/????(@)x$(~???1?/????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??e?c]??!c???XP@)<Nё\???1P?C???#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??&S??!????W?2@)46<?R??1? ?/?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ?	???!`Z;??7@)P?s???16?u0?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??A?f??!?`A
p?@)??A?f??1?`A
p?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?P?4??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??X?O??r~:T=????W?2ı?!??H.???	!       "	!       *	!       2$	??Q:*????
?yys??jM??St??!O??e?c??:	!       B	!       J$	??n,R?????????ZӼ???!???1段?R	!       Z$	??n,R?????????ZӼ???!???1段?JCPU_ONLYY?P?4??@b 